"""Dataset downloader for industry-standard conversational datasets.

Downloads real datasets from HuggingFace:
- LoCoMo: Long-context memory benchmark (Stanford SNAP)
- Persona-Chat: Persona-based conversations (Facebook)
- MultiWOZ: Multi-domain task-oriented dialogues

These are production-grade datasets used by researchers worldwide.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    turn_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """A complete conversation session."""

    session_id: str
    user_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    persona: list[str] = field(default_factory=list)  # For Persona-Chat
    domain: str | None = None  # For MultiWOZ
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def total_tokens_estimate(self) -> int:
        return sum(len(t.content.split()) for t in self.turns)


@dataclass
class DatasetSegment:
    """A segment of a dataset ready for enrichment."""

    dataset_name: str
    segment_id: str
    sessions: list[ConversationSession]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_turns(self) -> int:
        return sum(s.total_turns for s in self.sessions)


class DatasetDownloader:
    """Downloads and segments real conversational datasets."""

    def __init__(self, cache_dir: str | Path | None = None):
        """Initialize the downloader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = (
            Path(cache_dir) if cache_dir else Path.home() / ".mindcore" / "real_datasets"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._datasets_lib = None

    def _ensure_datasets_lib(self) -> None:
        """Ensure HuggingFace datasets library is available."""
        if self._datasets_lib is None:
            try:
                import datasets

                self._datasets_lib = datasets
            except ImportError:
                raise ImportError(
                    "HuggingFace datasets library required. Install with: pip install datasets"
                )

    def download_locomo(
        self,
        max_sessions: int = 100,
        max_turns_per_session: int = 50,
    ) -> DatasetSegment:
        """Download LoCoMo dataset from HuggingFace.

        LoCoMo (Long-Context Conversational Memory) is a benchmark
        for evaluating long-term memory in conversational AI.

        Source: https://github.com/snap-stanford/locomo
        Paper: https://arxiv.org/abs/2402.17753

        Args:
            max_sessions: Maximum number of sessions to download
            max_turns_per_session: Maximum turns per session

        Returns:
            DatasetSegment with LoCoMo conversations
        """
        self._ensure_datasets_lib()
        logger.info("Downloading LoCoMo dataset...")

        try:
            # LoCoMo is available on HuggingFace
            dataset = self._datasets_lib.load_dataset(
                "snap-stanford/locomo",
                split="test",
                cache_dir=str(self.cache_dir / "locomo"),
            )
        except Exception as e:
            logger.warning(f"Could not download LoCoMo from HuggingFace: {e}")
            logger.info("Falling back to sample LoCoMo-style data...")
            return self._generate_locomo_sample(max_sessions, max_turns_per_session)

        sessions = []
        for idx, example in enumerate(dataset):
            if idx >= max_sessions:
                break

            session = self._parse_locomo_example(example, idx, max_turns_per_session)
            if session:
                sessions.append(session)

        logger.info(f"Downloaded {len(sessions)} LoCoMo sessions")

        return DatasetSegment(
            dataset_name="locomo",
            segment_id=f"locomo_segment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={
                "source": "snap-stanford/locomo",
                "source_url": "https://huggingface.co/datasets/snap-stanford/locomo",
                "paper": "https://arxiv.org/abs/2402.17753",
                "max_sessions": max_sessions,
                "max_turns_per_session": max_turns_per_session,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _parse_locomo_example(
        self,
        example: dict,
        session_idx: int,
        max_turns: int,
    ) -> ConversationSession | None:
        """Parse a LoCoMo example into a ConversationSession."""
        try:
            # LoCoMo format: conversation field contains turns
            conversation = example.get("conversation", example.get("dialogue", []))
            if not conversation:
                return None

            turns = []
            for turn_idx, turn in enumerate(conversation[:max_turns]):
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("speaker", "user"))
                    content = turn.get("content", turn.get("text", turn.get("utterance", "")))
                elif isinstance(turn, str):
                    # Alternate user/assistant
                    role = "user" if turn_idx % 2 == 0 else "assistant"
                    content = turn
                else:
                    continue

                # Normalize role
                role = "user" if role.lower() in ["user", "human", "person1"] else "assistant"

                turns.append(
                    ConversationTurn(
                        role=role,
                        content=content,
                        turn_index=turn_idx,
                        metadata={
                            "original_speaker": turn.get("speaker")
                            if isinstance(turn, dict)
                            else None
                        },
                    )
                )

            if not turns:
                return None

            # Extract user info if available
            user_id = example.get("user_id", f"locomo_user_{session_idx}")

            return ConversationSession(
                session_id=f"locomo_{session_idx}",
                user_id=user_id,
                turns=turns,
                metadata={
                    "dataset": "locomo",
                    "original_id": example.get("id", session_idx),
                    "context": example.get("context", {}),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse LoCoMo example {session_idx}: {e}")
            return None

    def _generate_locomo_sample(
        self,
        max_sessions: int,
        max_turns_per_session: int,
    ) -> DatasetSegment:
        """Generate sample data in LoCoMo style when download fails."""
        # Realistic long-term memory conversation patterns from LoCoMo paper
        sample_conversations = [
            {
                "user_id": "alice_tech",
                "persona": ["software engineer", "prefers Python", "works remotely"],
                "turns": [
                    ("user", "Hi! I'm Alice, a software engineer. I mostly work with Python."),
                    (
                        "assistant",
                        "Nice to meet you, Alice! Python is a great choice for software development.",
                    ),
                    ("user", "I've been working remotely for about 3 years now."),
                    (
                        "assistant",
                        "Remote work has become quite popular. How do you find the experience?",
                    ),
                    ("user", "I love it! I've set up a nice home office with dual monitors."),
                    ("assistant", "A good setup makes all the difference for productivity."),
                    (
                        "user",
                        "Exactly. I also prefer dark mode for everything - easier on the eyes.",
                    ),
                    (
                        "assistant",
                        "Dark mode is definitely popular among developers for reducing eye strain.",
                    ),
                    ("user", "My favorite framework is FastAPI for building APIs."),
                    (
                        "assistant",
                        "FastAPI is excellent - great performance and automatic documentation.",
                    ),
                    ("user", "I usually start work around 9am PST."),
                    (
                        "assistant",
                        "Good to know your schedule. That's a common start time for West Coast developers.",
                    ),
                ],
            },
            {
                "user_id": "bob_data",
                "persona": ["data scientist", "uses R and Python", "interested in ML"],
                "turns": [
                    ("user", "Hello, I'm Bob. I work as a data scientist at a healthcare company."),
                    ("assistant", "Hello Bob! Healthcare data science must be fascinating work."),
                    (
                        "user",
                        "It is! I primarily use Python for machine learning and R for statistics.",
                    ),
                    (
                        "assistant",
                        "That's a powerful combination. Python for ML and R for statistical analysis.",
                    ),
                    (
                        "user",
                        "I'm particularly interested in medical imaging and diagnostic models.",
                    ),
                    (
                        "assistant",
                        "Medical imaging ML is a rapidly advancing field with real impact.",
                    ),
                    ("user", "I prefer working with Jupyter notebooks for exploratory analysis."),
                    ("assistant", "Jupyter is great for iterative exploration and documentation."),
                    ("user", "My team meets every Tuesday at 10am for our weekly sync."),
                    ("assistant", "Regular syncs help keep everyone aligned on projects."),
                    ("user", "I'm currently learning about transformer architectures for vision."),
                    ("assistant", "Vision transformers are transforming medical imaging analysis."),
                ],
            },
            {
                "user_id": "carol_design",
                "persona": ["UX designer", "uses Figma", "advocates for accessibility"],
                "turns": [
                    ("user", "Hi there! I'm Carol, a UX designer focusing on accessible design."),
                    (
                        "assistant",
                        "Welcome Carol! Accessibility is such an important aspect of design.",
                    ),
                    ("user", "I use Figma as my primary design tool."),
                    ("assistant", "Figma is fantastic for collaborative design work."),
                    ("user", "I always ensure our designs meet WCAG 2.1 AA standards."),
                    ("assistant", "Meeting WCAG standards ensures your designs work for everyone."),
                    ("user", "Color contrast and keyboard navigation are my top priorities."),
                    ("assistant", "Both are crucial for users with visual or motor impairments."),
                    (
                        "user",
                        "I work closely with the development team to ensure proper implementation.",
                    ),
                    (
                        "assistant",
                        "Designer-developer collaboration is key to accessible products.",
                    ),
                    ("user", "I prefer morning meetings before 11am when I'm most creative."),
                    ("assistant", "It's smart to protect your peak creative hours."),
                ],
            },
            {
                "user_id": "david_devops",
                "persona": ["DevOps engineer", "uses Kubernetes", "automates everything"],
                "turns": [
                    ("user", "Hey, I'm David. I handle DevOps and infrastructure."),
                    ("assistant", "Hi David! DevOps is critical for modern software delivery."),
                    ("user", "I manage our Kubernetes clusters across three cloud providers."),
                    ("assistant", "Multi-cloud K8s management requires serious expertise."),
                    ("user", "I believe in infrastructure as code - everything in Terraform."),
                    (
                        "assistant",
                        "IaC with Terraform ensures reproducible and auditable infrastructure.",
                    ),
                    ("user", "We use ArgoCD for GitOps-based deployments."),
                    ("assistant", "GitOps with ArgoCD provides excellent deployment automation."),
                    ("user", "I'm on-call every third week for production issues."),
                    ("assistant", "On-call rotations are essential for 24/7 system reliability."),
                    ("user", "I prefer to receive alerts via Slack rather than email."),
                    ("assistant", "Slack alerts are more immediate for time-sensitive issues."),
                ],
            },
            {
                "user_id": "eva_pm",
                "persona": ["product manager", "uses Notion", "data-driven decisions"],
                "turns": [
                    ("user", "Hello! I'm Eva, a product manager for a fintech startup."),
                    ("assistant", "Hi Eva! Fintech PM must be an exciting and fast-paced role."),
                    ("user", "It definitely is! I manage our mobile banking features."),
                    ("assistant", "Mobile banking is transforming how people manage finances."),
                    ("user", "I track all our product specs and roadmap in Notion."),
                    ("assistant", "Notion is great for keeping product documentation organized."),
                    ("user", "I make decisions based on user analytics and A/B testing results."),
                    ("assistant", "Data-driven decision making leads to better product outcomes."),
                    ("user", "My team does sprint planning every other Monday."),
                    ("assistant", "Bi-weekly sprints give good balance of planning and execution."),
                    ("user", "I prefer async communication for non-urgent matters."),
                    ("assistant", "Async communication respects everyone's focus time."),
                ],
            },
        ]

        sessions = []
        for idx, conv in enumerate(sample_conversations[:max_sessions]):
            turns = []
            for turn_idx, (role, content) in enumerate(conv["turns"][:max_turns_per_session]):
                turns.append(
                    ConversationTurn(
                        role=role,
                        content=content,
                        turn_index=turn_idx,
                    )
                )

            sessions.append(
                ConversationSession(
                    session_id=f"locomo_sample_{idx}",
                    user_id=conv["user_id"],
                    turns=turns,
                    persona=conv["persona"],
                    metadata={"dataset": "locomo_sample", "synthetic": True},
                )
            )

        return DatasetSegment(
            dataset_name="locomo_sample",
            segment_id=f"locomo_sample_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={
                "source": "synthetic (modeled after LoCoMo)",
                "paper": "https://arxiv.org/abs/2402.17753",
                "note": "Sample data - install 'datasets' library for real LoCoMo data",
            },
        )

    def download_persona_chat(
        self,
        max_sessions: int = 100,
        max_turns_per_session: int = 20,
    ) -> DatasetSegment:
        """Download Persona-Chat dataset from HuggingFace.

        Persona-Chat contains conversations where each participant
        has a defined persona they maintain throughout the dialogue.

        Source: https://huggingface.co/datasets/bavard/personachat_truecased
        Paper: https://arxiv.org/abs/1801.07243

        Args:
            max_sessions: Maximum number of sessions to download
            max_turns_per_session: Maximum turns per session

        Returns:
            DatasetSegment with Persona-Chat conversations
        """
        self._ensure_datasets_lib()
        logger.info("Downloading Persona-Chat dataset...")

        try:
            # Try the truecased version first (better quality)
            dataset = self._datasets_lib.load_dataset(
                "bavard/personachat_truecased",
                split="train",
                cache_dir=str(self.cache_dir / "persona_chat"),
            )
        except Exception:
            try:
                # Fallback to original
                dataset = self._datasets_lib.load_dataset(
                    "AlekseyKorshuk/persona-chat",
                    split="train",
                    cache_dir=str(self.cache_dir / "persona_chat"),
                )
            except Exception as e:
                logger.warning(f"Could not download Persona-Chat: {e}")
                return self._generate_persona_chat_sample(max_sessions, max_turns_per_session)

        sessions = []
        for idx, example in enumerate(dataset):
            if idx >= max_sessions:
                break

            session = self._parse_persona_chat_example(example, idx, max_turns_per_session)
            if session:
                sessions.append(session)

        logger.info(f"Downloaded {len(sessions)} Persona-Chat sessions")

        return DatasetSegment(
            dataset_name="persona_chat",
            segment_id=f"persona_chat_segment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={
                "source": "bavard/personachat_truecased",
                "paper": "https://arxiv.org/abs/1801.07243",
                "max_sessions": max_sessions,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _parse_persona_chat_example(
        self,
        example: dict,
        session_idx: int,
        max_turns: int,
    ) -> ConversationSession | None:
        """Parse a Persona-Chat example into a ConversationSession."""
        try:
            # Extract persona
            persona = example.get("personality", example.get("persona", []))
            if isinstance(persona, str):
                persona = [persona]

            # Extract conversation history
            history = example.get("history", example.get("utterances", []))
            if not history:
                # Try candidates format
                candidates = example.get("candidates", [])
                if candidates:
                    history = example.get("history", [])

            turns = []
            if isinstance(history, list):
                for turn_idx, utterance in enumerate(history[:max_turns]):
                    if isinstance(utterance, str):
                        role = "user" if turn_idx % 2 == 0 else "assistant"
                        content = utterance
                    elif isinstance(utterance, dict):
                        role = utterance.get("role", "user" if turn_idx % 2 == 0 else "assistant")
                        content = utterance.get("text", utterance.get("content", ""))
                    else:
                        continue

                    turns.append(
                        ConversationTurn(
                            role=role,
                            content=content,
                            turn_index=turn_idx,
                        )
                    )

            if not turns:
                return None

            return ConversationSession(
                session_id=f"persona_chat_{session_idx}",
                user_id=f"persona_user_{session_idx}",
                turns=turns,
                persona=persona,
                metadata={
                    "dataset": "persona_chat",
                    "original_idx": session_idx,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse Persona-Chat example {session_idx}: {e}")
            return None

    def _generate_persona_chat_sample(
        self,
        max_sessions: int,
        max_turns_per_session: int,
    ) -> DatasetSegment:
        """Generate sample Persona-Chat style data."""
        sample_personas = [
            {
                "persona": [
                    "I am a vegetarian",
                    "I love hiking in the mountains",
                    "I work as a nurse",
                ],
                "turns": [
                    ("user", "Hi! What do you do for a living?"),
                    (
                        "assistant",
                        "I work as a nurse at a local hospital. It's challenging but rewarding.",
                    ),
                    ("user", "That's wonderful! Do you have any hobbies?"),
                    (
                        "assistant",
                        "I love hiking! There's nothing like being in the mountains on a weekend.",
                    ),
                    ("user", "What about food preferences?"),
                    (
                        "assistant",
                        "I'm a vegetarian. I find it healthier and better for the environment.",
                    ),
                ],
            },
            {
                "persona": [
                    "I play guitar in a band",
                    "I'm studying computer science",
                    "I have two cats",
                ],
                "turns": [
                    ("user", "Do you have any pets?"),
                    ("assistant", "Yes! I have two cats. They're always entertaining me."),
                    ("user", "What are you studying?"),
                    (
                        "assistant",
                        "I'm studying computer science. It's fascinating how technology evolves.",
                    ),
                    ("user", "Any musical interests?"),
                    ("assistant", "I play guitar in a band! We mostly do rock covers on weekends."),
                ],
            },
            {
                "persona": [
                    "I'm a professional chef",
                    "I travel frequently for work",
                    "I speak three languages",
                ],
                "turns": [
                    ("user", "What's your profession?"),
                    ("assistant", "I'm a professional chef. I specialize in French cuisine."),
                    ("user", "Do you travel much?"),
                    (
                        "assistant",
                        "Quite a bit! My work takes me to different cities for pop-up events.",
                    ),
                    ("user", "That's exciting! Do you speak other languages?"),
                    (
                        "assistant",
                        "I speak three languages - English, French, and Spanish. It helps when traveling.",
                    ),
                ],
            },
        ]

        sessions = []
        for idx, data in enumerate(sample_personas[:max_sessions]):
            turns = []
            for turn_idx, (role, content) in enumerate(data["turns"][:max_turns_per_session]):
                turns.append(
                    ConversationTurn(
                        role=role,
                        content=content,
                        turn_index=turn_idx,
                    )
                )

            sessions.append(
                ConversationSession(
                    session_id=f"persona_sample_{idx}",
                    user_id=f"persona_user_{idx}",
                    turns=turns,
                    persona=data["persona"],
                    metadata={"synthetic": True},
                )
            )

        return DatasetSegment(
            dataset_name="persona_chat_sample",
            segment_id=f"persona_sample_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={"source": "synthetic", "note": "Sample data"},
        )

    def download_multiwoz(
        self,
        max_sessions: int = 100,
        max_turns_per_session: int = 30,
    ) -> DatasetSegment:
        """Download MultiWOZ dataset from HuggingFace.

        MultiWOZ is a multi-domain task-oriented dialogue dataset
        spanning hotel, restaurant, attraction, taxi, and train domains.

        Source: https://huggingface.co/datasets/multi_woz_v22
        Paper: https://arxiv.org/abs/1810.00278

        Args:
            max_sessions: Maximum number of sessions to download
            max_turns_per_session: Maximum turns per session

        Returns:
            DatasetSegment with MultiWOZ conversations
        """
        self._ensure_datasets_lib()
        logger.info("Downloading MultiWOZ dataset...")

        try:
            dataset = self._datasets_lib.load_dataset(
                "multi_woz_v22",
                split="train",
                cache_dir=str(self.cache_dir / "multiwoz"),
            )
        except Exception as e:
            logger.warning(f"Could not download MultiWOZ: {e}")
            return self._generate_multiwoz_sample(max_sessions, max_turns_per_session)

        sessions = []
        for idx, example in enumerate(dataset):
            if idx >= max_sessions:
                break

            session = self._parse_multiwoz_example(example, idx, max_turns_per_session)
            if session:
                sessions.append(session)

        logger.info(f"Downloaded {len(sessions)} MultiWOZ sessions")

        return DatasetSegment(
            dataset_name="multiwoz",
            segment_id=f"multiwoz_segment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={
                "source": "multi_woz_v22",
                "paper": "https://arxiv.org/abs/1810.00278",
                "domains": ["hotel", "restaurant", "attraction", "taxi", "train"],
                "max_sessions": max_sessions,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _parse_multiwoz_example(
        self,
        example: dict,
        session_idx: int,
        max_turns: int,
    ) -> ConversationSession | None:
        """Parse a MultiWOZ example into a ConversationSession."""
        try:
            turns_data = example.get("turns", {})
            utterances = turns_data.get("utterance", [])
            speakers = turns_data.get("speaker", [])

            if not utterances:
                return None

            turns = []
            for turn_idx, (utterance, speaker) in enumerate(
                zip(
                    utterances[:max_turns],
                    speakers[:max_turns] if speakers else [0, 1] * (len(utterances) // 2 + 1),
                    strict=False,
                )
            ):
                role = "user" if speaker == 0 else "assistant"
                turns.append(
                    ConversationTurn(
                        role=role,
                        content=utterance,
                        turn_index=turn_idx,
                    )
                )

            if not turns:
                return None

            # Extract domain from services
            services = example.get("services", [])
            domain = services[0] if services else "general"

            return ConversationSession(
                session_id=f"multiwoz_{session_idx}",
                user_id=f"multiwoz_user_{session_idx}",
                turns=turns,
                domain=domain,
                metadata={
                    "dataset": "multiwoz",
                    "dialogue_id": example.get("dialogue_id", session_idx),
                    "services": services,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to parse MultiWOZ example {session_idx}: {e}")
            return None

    def _generate_multiwoz_sample(
        self,
        max_sessions: int,
        max_turns_per_session: int,
    ) -> DatasetSegment:
        """Generate sample MultiWOZ style data."""
        sample_dialogues = [
            {
                "domain": "hotel",
                "turns": [
                    ("user", "I need to book a hotel in Cambridge for this weekend."),
                    ("assistant", "I'd be happy to help. How many nights will you be staying?"),
                    ("user", "Two nights, Friday and Saturday. I need parking."),
                    (
                        "assistant",
                        "I found several hotels with parking. Do you have a price preference?",
                    ),
                    ("user", "Something moderate, not too expensive."),
                    (
                        "assistant",
                        "The Hamilton Lodge has parking and is moderately priced at $85/night.",
                    ),
                    ("user", "That sounds perfect. Can you book it for me?"),
                    (
                        "assistant",
                        "Booking confirmed for Hamilton Lodge, 2 nights. Reference: HT7829.",
                    ),
                ],
            },
            {
                "domain": "restaurant",
                "turns": [
                    ("user", "I'm looking for a nice Italian restaurant in the city center."),
                    (
                        "assistant",
                        "There are several Italian restaurants downtown. Any price range?",
                    ),
                    ("user", "Mid-range would be ideal. For dinner tonight at 7pm."),
                    (
                        "assistant",
                        "Bella Italia has excellent reviews and is mid-range. Table for how many?",
                    ),
                    ("user", "Table for 4, please."),
                    (
                        "assistant",
                        "Reserved! Table for 4 at Bella Italia, tonight 7pm. Reference: RS4521.",
                    ),
                ],
            },
            {
                "domain": "train",
                "turns": [
                    ("user", "I need a train from London to Manchester tomorrow morning."),
                    ("assistant", "What time would you like to depart?"),
                    ("user", "Around 9am would be ideal."),
                    (
                        "assistant",
                        "There's a 9:15 train arriving at 11:45. First class or standard?",
                    ),
                    ("user", "Standard is fine. How much is it?"),
                    ("assistant", "Standard fare is $45. Shall I book it?"),
                    ("user", "Yes please, one ticket."),
                    (
                        "assistant",
                        "Booked! London to Manchester, 9:15am tomorrow. Reference: TR8834.",
                    ),
                ],
            },
        ]

        sessions = []
        for idx, dialogue in enumerate(sample_dialogues[:max_sessions]):
            turns = []
            for turn_idx, (role, content) in enumerate(dialogue["turns"][:max_turns_per_session]):
                turns.append(
                    ConversationTurn(
                        role=role,
                        content=content,
                        turn_index=turn_idx,
                    )
                )

            sessions.append(
                ConversationSession(
                    session_id=f"multiwoz_sample_{idx}",
                    user_id=f"multiwoz_user_{idx}",
                    turns=turns,
                    domain=dialogue["domain"],
                    metadata={"synthetic": True},
                )
            )

        return DatasetSegment(
            dataset_name="multiwoz_sample",
            segment_id=f"multiwoz_sample_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            sessions=sessions,
            metadata={"source": "synthetic", "note": "Sample data"},
        )

    def download_all(
        self,
        max_sessions_per_dataset: int = 50,
        max_turns_per_session: int = 30,
    ) -> list[DatasetSegment]:
        """Download all available datasets.

        Args:
            max_sessions_per_dataset: Max sessions per dataset
            max_turns_per_session: Max turns per session

        Returns:
            List of DatasetSegments
        """
        segments = []

        logger.info("Downloading all datasets...")

        # Download each dataset
        segments.append(self.download_locomo(max_sessions_per_dataset, max_turns_per_session))
        segments.append(self.download_persona_chat(max_sessions_per_dataset, max_turns_per_session))
        segments.append(self.download_multiwoz(max_sessions_per_dataset, max_turns_per_session))

        total_sessions = sum(s.total_sessions for s in segments)
        total_turns = sum(s.total_turns for s in segments)

        logger.info(
            f"Downloaded {len(segments)} datasets with {total_sessions} sessions and {total_turns} turns"
        )

        return segments

    def save_segment(self, segment: DatasetSegment, path: Path | str) -> None:
        """Save a dataset segment to JSON file.

        Args:
            segment: DatasetSegment to save
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset_name": segment.dataset_name,
            "segment_id": segment.segment_id,
            "metadata": segment.metadata,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "user_id": s.user_id,
                    "persona": s.persona,
                    "domain": s.domain,
                    "metadata": s.metadata,
                    "turns": [
                        {
                            "role": t.role,
                            "content": t.content,
                            "turn_index": t.turn_index,
                            "metadata": t.metadata,
                        }
                        for t in s.turns
                    ],
                }
                for s in segment.sessions
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved segment to {path}")

    def load_segment(self, path: Path | str) -> DatasetSegment:
        """Load a dataset segment from JSON file.

        Args:
            path: Input file path

        Returns:
            DatasetSegment
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        sessions = []
        for s_data in data["sessions"]:
            turns = [
                ConversationTurn(
                    role=t["role"],
                    content=t["content"],
                    turn_index=t["turn_index"],
                    metadata=t.get("metadata", {}),
                )
                for t in s_data["turns"]
            ]

            sessions.append(
                ConversationSession(
                    session_id=s_data["session_id"],
                    user_id=s_data["user_id"],
                    turns=turns,
                    persona=s_data.get("persona", []),
                    domain=s_data.get("domain"),
                    metadata=s_data.get("metadata", {}),
                )
            )

        return DatasetSegment(
            dataset_name=data["dataset_name"],
            segment_id=data["segment_id"],
            sessions=sessions,
            metadata=data.get("metadata", {}),
        )
