"""Encryption at rest module for Mindcore.

Provides field-level encryption for sensitive memory content using
the cryptography library with Fernet symmetric encryption.

Requirements:
    pip install cryptography

Example:
    from mindcore.enterprise import EncryptionConfig, FieldEncryptor

    # From environment variable (recommended)
    config = EncryptionConfig(key_from_env="MINDCORE_ENCRYPTION_KEY")
    encryptor = FieldEncryptor(config)

    # Or generate a new key
    key = FieldEncryptor.generate_key()
    encryptor = FieldEncryptor(EncryptionConfig(key=key))

    # Encrypt/decrypt memory content
    encrypted = encryptor.encrypt("sensitive user data")
    decrypted = encryptor.decrypt(encrypted)

    # Key rotation
    rotator = KeyRotator([old_key, new_key])
    rotator.rotate_memory(memory)

Security Best Practices (from cryptography.io):
    - Use at least 1,200,000 PBKDF2 iterations for key derivation
    - Store keys securely (env vars, secrets manager, HSM)
    - Implement key rotation
    - Don't store plaintext alongside encrypted data

References:
    - https://cryptography.io/en/latest/fernet/
    - https://www.comparitech.com/blog/information-security/what-is-fernet/
"""

from __future__ import annotations

import base64
import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Any


# Type hints for optional dependency
try:
    from cryptography.fernet import Fernet, InvalidToken, MultiFernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    Fernet = None
    MultiFernet = None


class EncryptionError(Exception):
    """Base exception for encryption errors.

    Attributes:
        message: Error message
        field: Field that failed (if applicable)
        operation: Operation that failed (encrypt/decrypt)
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        operation: str | None = None,
    ):
        super().__init__(message)
        self.field = field
        self.operation = operation


@dataclass
class EncryptionConfig:
    """Configuration for field-level encryption.

    Attributes:
        key: Encryption key (base64-encoded Fernet key)
        key_from_env: Environment variable containing the key
        key_from_file: Path to file containing the key
        password: Password for key derivation (less secure than raw key)
        salt: Salt for PBKDF2 key derivation (required with password)
        kdf_iterations: PBKDF2 iteration count (default: 1,200,000 per Django 2025)
        encrypt_fields: Fields to encrypt (default: ["content"])
        encrypted_prefix: Prefix for encrypted values
        rotation_keys: Additional keys for decryption during rotation

    Security Notes:
        - Prefer key_from_env over hardcoded keys
        - Use secrets manager (Vault, AWS SM) for production
        - kdf_iterations should be >= 1,200,000 (Django recommendation Jan 2025)

    Example:
        # From environment (recommended)
        config = EncryptionConfig(key_from_env="MINDCORE_ENCRYPTION_KEY")

        # With password-based key derivation
        config = EncryptionConfig(
            password="strong-password",
            salt="unique-salt-per-deployment",
            kdf_iterations=1_200_000,
        )
    """

    key: str | None = None
    key_from_env: str | None = None
    key_from_file: str | None = None
    password: str | None = None
    salt: str | None = None
    kdf_iterations: int = 1_200_000  # Django recommendation as of Jan 2025
    encrypt_fields: list[str] = field(default_factory=lambda: ["content"])
    encrypted_prefix: str = "enc:v1:"
    rotation_keys: list[str] = field(default_factory=list)


class FieldEncryptor:
    """Field-level encryption for memory content.

    Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256)
    for authenticated encryption.

    Example:
        # Basic usage
        encryptor = FieldEncryptor(EncryptionConfig(
            key_from_env="MINDCORE_ENCRYPTION_KEY"
        ))

        # Encrypt content
        encrypted = encryptor.encrypt("sensitive data")
        print(encrypted)  # enc:v1:gAAA...

        # Decrypt content
        decrypted = encryptor.decrypt(encrypted)

        # Encrypt memory dict
        memory = {"content": "sensitive", "topics": ["personal"]}
        encrypted_memory = encryptor.encrypt_memory(memory)
        decrypted_memory = encryptor.decrypt_memory(encrypted_memory)

        # Check if value is encrypted
        if encryptor.is_encrypted(value):
            value = encryptor.decrypt(value)
    """

    def __init__(self, config: EncryptionConfig):
        """Initialize field encryptor.

        Args:
            config: Encryption configuration

        Raises:
            ImportError: If cryptography is not installed
            EncryptionError: If key configuration is invalid
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError(
                "cryptography library required for encryption. "
                "Install with: pip install cryptography"
            )

        self.config = config
        self._setup_encryption()

    def _setup_encryption(self) -> None:
        """Initialize Fernet encryption."""
        key = self._get_key()

        if not key:
            raise EncryptionError(
                "No encryption key configured. Provide key, key_from_env, "
                "key_from_file, or password+salt in EncryptionConfig."
            )

        # Create primary Fernet instance
        try:
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
        except Exception as e:
            raise EncryptionError(f"Invalid encryption key: {e}")

        # Setup MultiFernet for key rotation if rotation keys provided
        if self.config.rotation_keys:
            fernets = [self._fernet]
            for rot_key in self.config.rotation_keys:
                try:
                    fernets.append(
                        Fernet(rot_key.encode() if isinstance(rot_key, str) else rot_key)
                    )
                except Exception:
                    pass  # Skip invalid rotation keys
            self._multi_fernet = MultiFernet(fernets)
        else:
            self._multi_fernet = None

    def _get_key(self) -> str | None:
        """Get encryption key from configured source.

        Returns:
            Base64-encoded Fernet key, or None if not configured
        """
        # Direct key
        if self.config.key:
            return self.config.key

        # From environment variable
        if self.config.key_from_env:
            key = os.environ.get(self.config.key_from_env)
            if key:
                return key

        # From file
        if self.config.key_from_file:
            try:
                with open(self.config.key_from_file) as f:
                    return f.read().strip()
            except FileNotFoundError:
                pass

        # Derive from password
        if self.config.password and self.config.salt:
            return self._derive_key(self.config.password, self.config.salt)

        return None

    def _derive_key(self, password: str, salt: str) -> str:
        """Derive a Fernet key from password using PBKDF2.

        Args:
            password: Password to derive from
            salt: Salt for key derivation

        Returns:
            Base64-encoded Fernet key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode(),
            iterations=self.config.kdf_iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode()

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key.

        Returns:
            Base64-encoded Fernet key

        Example:
            key = FieldEncryptor.generate_key()
            # Store this securely (env var, secrets manager)
            config = EncryptionConfig(key=key)
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required")
        return Fernet.generate_key().decode()

    @staticmethod
    def generate_salt(length: int = 32) -> str:
        """Generate a cryptographically secure salt.

        Args:
            length: Length of salt in bytes

        Returns:
            Hex-encoded salt string
        """
        return secrets.token_hex(length)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string value.

        Args:
            plaintext: Value to encrypt

        Returns:
            Encrypted value with prefix

        Raises:
            EncryptionError: If encryption fails
        """
        if not plaintext:
            return plaintext

        try:
            encrypted = self._fernet.encrypt(plaintext.encode())
            return f"{self.config.encrypted_prefix}{encrypted.decode()}"
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}", operation="encrypt")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt an encrypted value.

        Args:
            ciphertext: Encrypted value (with or without prefix)

        Returns:
            Decrypted plaintext

        Raises:
            EncryptionError: If decryption fails
        """
        if not ciphertext:
            return ciphertext

        # Remove prefix if present
        if ciphertext.startswith(self.config.encrypted_prefix):
            ciphertext = ciphertext[len(self.config.encrypted_prefix) :]

        try:
            # Use MultiFernet if available (for key rotation)
            fernet = self._multi_fernet or self._fernet
            decrypted = fernet.decrypt(ciphertext.encode())
            return decrypted.decode()
        except InvalidToken:
            raise EncryptionError("Decryption failed: invalid token or key", operation="decrypt")
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}", operation="decrypt")

    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted.

        Args:
            value: Value to check

        Returns:
            True if value appears to be encrypted
        """
        if not value or not isinstance(value, str):
            return False
        return value.startswith(self.config.encrypted_prefix)

    def encrypt_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Encrypt configured fields in a memory dict.

        Args:
            memory: Memory dictionary

        Returns:
            Memory with encrypted fields

        Example:
            memory = {"content": "sensitive", "topics": ["personal"]}
            encrypted = encryptor.encrypt_memory(memory)
            # encrypted["content"] = "enc:v1:gAAA..."
        """
        result = memory.copy()

        for field_name in self.config.encrypt_fields:
            if result.get(field_name):
                value = result[field_name]

                # Skip if already encrypted
                if isinstance(value, str) and self.is_encrypted(value):
                    continue

                # Convert non-strings to JSON
                if not isinstance(value, str):
                    value = json.dumps(value)

                result[field_name] = self.encrypt(value)

        # Add encryption metadata
        result["_encrypted"] = True
        result["_encryption_version"] = "v1"

        return result

    def decrypt_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Decrypt configured fields in a memory dict.

        Args:
            memory: Encrypted memory dictionary

        Returns:
            Memory with decrypted fields

        Example:
            decrypted = encryptor.decrypt_memory(encrypted_memory)
        """
        result = memory.copy()

        for field_name in self.config.encrypt_fields:
            if result.get(field_name):
                value = result[field_name]

                if isinstance(value, str) and self.is_encrypted(value):
                    result[field_name] = self.decrypt(value)

        # Remove encryption metadata
        result.pop("_encrypted", None)
        result.pop("_encryption_version", None)

        return result

    def rotate_key(self, ciphertext: str, new_key: str | None = None) -> str:
        """Re-encrypt a value with the current (or new) key.

        Useful for key rotation: decrypt with old key, encrypt with new.

        Args:
            ciphertext: Encrypted value
            new_key: Optional new key (uses current key if not provided)

        Returns:
            Re-encrypted value
        """
        # Decrypt with current keys (including rotation keys)
        plaintext = self.decrypt(ciphertext)

        # Re-encrypt with primary key (or new key)
        if new_key:
            new_fernet = Fernet(new_key.encode() if isinstance(new_key, str) else new_key)
            encrypted = new_fernet.encrypt(plaintext.encode())
            return f"{self.config.encrypted_prefix}{encrypted.decode()}"

        return self.encrypt(plaintext)


class KeyRotator:
    """Key rotation utility for encrypted memories.

    Supports rotating encryption keys across all memories while
    maintaining the ability to decrypt with old keys.

    Example:
        # Setup rotator with old and new keys
        rotator = KeyRotator(
            keys=["old-key-base64", "new-key-base64"],
            primary_key_index=1,  # Use new key for encryption
        )

        # Rotate a single memory
        rotated_memory = rotator.rotate_memory(memory)

        # Rotate all memories (generator for memory efficiency)
        for rotated in rotator.rotate_memories(memories):
            storage.update(rotated)
    """

    def __init__(
        self,
        keys: list[str],
        primary_key_index: int = 0,
        encrypt_fields: list[str] | None = None,
    ):
        """Initialize key rotator.

        Args:
            keys: List of encryption keys (base64-encoded)
            primary_key_index: Index of key to use for encryption
            encrypt_fields: Fields to encrypt (default: ["content"])
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise ImportError("cryptography library required")

        if not keys:
            raise ValueError("At least one key required")

        self.keys = keys
        self.primary_key_index = primary_key_index
        self.encrypt_fields = encrypt_fields or ["content"]

        # Create encryptor with primary key and all keys for decryption
        primary_key = keys[primary_key_index]
        rotation_keys = [k for i, k in enumerate(keys) if i != primary_key_index]

        self._encryptor = FieldEncryptor(
            EncryptionConfig(
                key=primary_key,
                rotation_keys=rotation_keys,
                encrypt_fields=self.encrypt_fields,
            )
        )

    def rotate_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Rotate encryption key for a single memory.

        Args:
            memory: Memory with encrypted fields

        Returns:
            Memory re-encrypted with primary key
        """
        # Decrypt with any key
        decrypted = self._encryptor.decrypt_memory(memory)

        # Re-encrypt with primary key
        return self._encryptor.encrypt_memory(decrypted)

    def rotate_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rotate encryption keys for multiple memories.

        Args:
            memories: List of memories

        Yields:
            Re-encrypted memories
        """
        for memory in memories:
            yield self.rotate_memory(memory)

    def verify_rotation(
        self,
        original: dict[str, Any],
        rotated: dict[str, Any],
    ) -> bool:
        """Verify that rotation preserved data integrity.

        Args:
            original: Original encrypted memory
            rotated: Rotated encrypted memory

        Returns:
            True if decrypted content matches
        """
        original_decrypted = self._encryptor.decrypt_memory(original)
        rotated_decrypted = self._encryptor.decrypt_memory(rotated)

        for field_name in self.encrypt_fields:
            if original_decrypted.get(field_name) != rotated_decrypted.get(field_name):
                return False

        return True
