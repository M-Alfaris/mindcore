"""
Custom Prompts Example

This example demonstrates how to customize AI agent prompts in Mindcore.

Three methods for customization:
1. Edit mindcore/prompts.py directly
2. Use YAML configuration file
3. Pass prompts programmatically to agents

Use cases:
- Domain-specific metadata extraction (e-commerce, healthcare, legal)
- Specialized context assembly (customer support, technical docs)
- Multi-language support
- Brand voice and tone customization
"""

import os
import tempfile
from mindcore.prompts import (
    ENRICHMENT_SYSTEM_PROMPT,
    CONTEXT_ASSEMBLY_SYSTEM_PROMPT,
    get_enrichment_prompt,
    get_context_assembly_prompt,
    load_custom_prompts
)


def example_1_view_default_prompts():
    """Example 1: View default prompts."""

    print("=" * 80)
    print("Example 1: Default Mindcore Prompts")
    print("=" * 80)
    print()

    print("ENRICHMENT SYSTEM PROMPT:")
    print("-" * 80)
    print(ENRICHMENT_SYSTEM_PROMPT)
    print()

    print("CONTEXT ASSEMBLY SYSTEM PROMPT:")
    print("-" * 80)
    print(CONTEXT_ASSEMBLY_SYSTEM_PROMPT)
    print()


def example_2_formatted_prompts():
    """Example 2: Using prompt formatting functions."""

    print("=" * 80)
    print("Example 2: Formatted Prompts")
    print("=" * 80)
    print()

    # Enrichment prompt
    print("Enrichment User Prompt:")
    print("-" * 80)
    enrichment_prompt = get_enrichment_prompt(
        role="user",
        text="This is urgent! We need to fix the critical bug in production ASAP."
    )
    print(enrichment_prompt)
    print()

    # Context assembly prompt
    print("Context Assembly User Prompt:")
    print("-" * 80)
    formatted_messages = """
1. [user] What's the status of the deployment?
   Topics: [deployment, status], Importance: 0.7

2. [assistant] The deployment is currently in progress. We're at 60% completion.
   Topics: [deployment, progress], Importance: 0.6

3. [user] Any issues so far?
   Topics: [deployment, issues], Importance: 0.8
"""
    context_prompt = get_context_assembly_prompt(
        formatted_messages=formatted_messages,
        query="deployment issues"
    )
    print(context_prompt)
    print()


def example_3_custom_prompts_for_ecommerce():
    """Example 3: Custom prompts for e-commerce platform."""

    print("=" * 80)
    print("Example 3: E-commerce Custom Prompts")
    print("=" * 80)
    print()

    # E-commerce enrichment prompt
    ecommerce_enrichment = """You are a metadata enrichment AI agent for an e-commerce customer support platform.

Your task is to analyze customer messages and extract structured metadata with a focus on e-commerce context.

For each message, return a JSON object with:

{
  "topics": ["product names", "categories", "order numbers", "issues"],
  "categories": ["inquiry", "complaint", "return_request", "order_status", "product_question", "shipping"],
  "importance": 0.0-1.0 (prioritize: payment > order issues > shipping > general questions),
  "sentiment": {
    "overall": "positive/negative/neutral",
    "score": 0.0-1.0
  },
  "intent": "check_order_status" | "request_refund" | "product_inquiry" | "shipping_question" | "complaint" | "return_request",
  "tags": ["product_names", "order_numbers", "urgent_indicators"],
  "entities": ["product names", "order IDs", "tracking numbers", "dates"],
  "key_phrases": ["important phrases from the message"],
  "order_id": "extracted order ID if present",
  "product_names": ["extracted product names"],
  "issue_type": "payment" | "shipping" | "product_quality" | "missing_item" | "wrong_item" | "other"
}

Priority rules:
- Payment issues: importance >= 0.9
- Order problems: importance >= 0.8
- Shipping questions: importance >= 0.6
- General inquiries: importance >= 0.4

Extract order IDs (format: ORD-XXXXX) and product names when present.
"""

    print("E-commerce Enrichment Prompt:")
    print("-" * 80)
    print(ecommerce_enrichment)
    print()

    # E-commerce context assembly prompt
    ecommerce_context = """You are a context assembly AI agent for e-commerce customer support.

You will receive:
1. Customer's message history with metadata
2. Current customer inquiry

Your task is to extract relevant context focusing on:
- Previous orders and their status
- Past product inquiries
- Shipping history
- Payment issues
- Return/refund requests

Return a JSON object:

{
  "assembled_context": "Concise summary of relevant order history, product interactions, and issues. Focus on what the support agent needs to know.",
  "key_points": [
    "Customer's previous orders and status",
    "Past issues and resolutions",
    "Relevant product interests",
    "Shipping preferences"
  ],
  "relevant_message_ids": ["IDs of most relevant messages"],
  "metadata": {
    "topics": ["main topics from history"],
    "sentiment": {
      "overall": "positive/negative/neutral",
      "trend": "improving/declining/stable"
    },
    "importance": 0.0-1.0,
    "customer_status": "new" | "repeat" | "vip" | "at_risk",
    "order_count": number,
    "has_open_issues": boolean
  }
}

Prioritize recent orders, unresolved issues, and payment-related history.
"""

    print("E-commerce Context Assembly Prompt:")
    print("-" * 80)
    print(ecommerce_context)
    print()


def example_4_custom_prompts_yaml():
    """Example 4: Loading custom prompts from YAML file."""

    print("=" * 80)
    print("Example 4: Custom Prompts via YAML Configuration")
    print("=" * 80)
    print()

    # Create example custom prompts YAML
    yaml_content = """# Custom Prompts for Healthcare Platform

enrichment_system_prompt: |
  You are a metadata enrichment AI agent for a healthcare platform.

  Analyze medical messages and extract structured metadata:

  {
    "topics": ["medical conditions", "symptoms", "treatments"],
    "categories": ["symptom_report", "medication_question", "appointment_request", "test_results", "emergency"],
    "importance": 0.0-1.0 (emergency > urgent symptoms > appointments > general questions),
    "sentiment": {
      "overall": "positive/negative/neutral",
      "score": 0.0-1.0
    },
    "intent": "report_symptoms" | "request_appointment" | "medication_inquiry" | "emergency" | "test_results",
    "tags": ["medical terms", "urgency indicators"],
    "entities": ["symptoms", "medications", "conditions"],
    "key_phrases": ["important medical phrases"],
    "urgency_level": "emergency" | "urgent" | "routine" | "follow_up"
  }

  IMPORTANT: Flag emergency keywords (chest pain, difficulty breathing, severe bleeding).

context_assembly_system_prompt: |
  You are a context assembly agent for healthcare.

  Analyze patient message history and current inquiry to provide relevant medical context.

  Return JSON:

  {
    "assembled_context": "Summary of relevant medical history, symptoms, medications, and past consultations.",
    "key_points": [
      "Previous symptoms and diagnoses",
      "Current medications",
      "Allergies and conditions",
      "Recent appointments"
    ],
    "relevant_message_ids": ["relevant message IDs"],
    "metadata": {
      "topics": ["medical topics"],
      "sentiment": {"overall": "...", "trend": "..."},
      "importance": 0.0-1.0,
      "has_emergency_history": boolean,
      "chronic_conditions": ["conditions"]
    }
  }

  Focus on medical accuracy and patient safety.
"""

    print("Example custom_prompts.yaml for Healthcare:")
    print("-" * 80)
    print(yaml_content)
    print()

    # Save to temp file and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Load custom prompts
        custom_prompts = load_custom_prompts(temp_path)

        print("Loaded custom prompts:")
        print("-" * 80)
        print(f"✓ enrichment_system_prompt ({len(custom_prompts['enrichment_system_prompt'])} chars)")
        print(f"✓ context_assembly_system_prompt ({len(custom_prompts['context_assembly_system_prompt'])} chars)")
        print()

        print("To use these prompts:")
        print("-" * 80)
        print("""
# In config.yaml:
prompts:
  custom_path: /path/to/custom_prompts.yaml

# Or programmatically:
from mindcore.prompts import load_custom_prompts
from mindcore import MetadataAgent

custom = load_custom_prompts("custom_prompts.yaml")

agent = MetadataAgent(
    api_key="your-key",
    system_prompt=custom["enrichment_system_prompt"]
)
""")

    finally:
        os.unlink(temp_path)


def example_5_specialized_prompts():
    """Example 5: Specialized prompts for different use cases."""

    print("=" * 80)
    print("Example 5: Specialized Prompts for Different Domains")
    print("=" * 80)
    print()

    # Technical Support
    print("1. Technical Support Prompt")
    print("-" * 80)
    tech_prompt = """You are a metadata enrichment AI for technical support tickets.

Focus on:
- Bug severity (critical, high, medium, low)
- Affected systems/components
- Error messages and stack traces
- User impact (all users, some users, single user)
- Environment (production, staging, development)

Extract technical entities: error codes, API endpoints, database tables, services.

Importance scoring:
- Production + critical: 1.0
- Production + high: 0.9
- Staging issues: 0.6
- Development: 0.4
"""
    print(tech_prompt)
    print()

    # Legal Document Analysis
    print("2. Legal Document Analysis Prompt")
    print("-" * 80)
    legal_prompt = """You are a metadata enrichment AI for legal document analysis.

Extract:
- Document type (contract, agreement, motion, brief, correspondence)
- Legal topics (intellectual property, employment, contracts, litigation)
- Parties involved
- Key dates and deadlines
- Legal precedents or citations
- Urgency indicators (filing deadlines, court dates)

Importance based on:
- Time-sensitive filings: 0.9-1.0
- Client correspondence: 0.7-0.8
- Research materials: 0.4-0.6
"""
    print(legal_prompt)
    print()

    # Educational Platform
    print("3. Educational Platform Prompt")
    print("-" * 80)
    edu_prompt = """You are a metadata enrichment AI for an educational platform.

Analyze student messages for:
- Topics: course subjects, assignments, deadlines
- Student sentiment (confused, frustrated, confident, curious)
- Question complexity (basic, intermediate, advanced)
- Help urgency (stuck, deadline soon, general inquiry)
- Learning indicators (understanding, misconception, progress)

Importance:
- Deadline < 24h: 0.9
- Student struggling: 0.8
- General questions: 0.5
"""
    print(edu_prompt)
    print()


def example_6_multilingual_prompts():
    """Example 6: Multi-language support."""

    print("=" * 80)
    print("Example 6: Multi-language Prompts")
    print("=" * 80)
    print()

    spanish_prompt = """Eres un agente de IA para enriquecimiento de metadatos.

Tu tarea es analizar mensajes en español y extraer metadatos estructurados.

Para cada mensaje, devuelve un objeto JSON con:

{
  "topics": ["lista de temas principales"],
  "categories": ["pregunta", "declaración", "comando", "código", "técnico", "casual", etc.],
  "importance": 0.0-1.0 (donde 1.0 es más importante),
  "sentiment": {
    "overall": "positivo/negativo/neutral",
    "score": 0.0-1.0
  },
  "intent": "hacer_pregunta" | "proporcionar_info" | "solicitar_acción" | "expresar_opinión" | "saludo",
  "tags": ["etiquetas o palabras clave relevantes"],
  "entities": ["nombres de personas, lugares, tecnologías, productos"],
  "key_phrases": ["frases importantes del mensaje"]
}

Sé conciso y preciso. Enfócate en extraer la información más relevante.
"""

    print("Spanish Enrichment Prompt:")
    print("-" * 80)
    print(spanish_prompt)
    print()

    print("Benefits of multi-language prompts:")
    print("  • Better accuracy for non-English content")
    print("  • Culturally appropriate context understanding")
    print("  • Improved sentiment analysis in native language")
    print()


def example_7_prompt_best_practices():
    """Example 7: Best practices for custom prompts."""

    print("=" * 80)
    print("Example 7: Best Practices for Custom Prompts")
    print("=" * 80)
    print()

    print("1. Be Specific")
    print("-" * 80)
    print("   ✓ Define exact JSON structure expected")
    print("   ✓ Specify importance scoring rules")
    print("   ✓ List specific entities to extract")
    print()

    print("2. Provide Examples")
    print("-" * 80)
    print("   ✓ Include sample inputs and outputs")
    print("   ✓ Show edge cases")
    print("   ✓ Demonstrate priority rules")
    print()

    print("3. Domain Vocabulary")
    print("-" * 80)
    print("   ✓ Use domain-specific terminology")
    print("   ✓ Define industry-specific categories")
    print("   ✓ Include relevant entity types")
    print()

    print("4. Clear Priorities")
    print("-" * 80)
    print("   ✓ Define importance scoring clearly")
    print("   ✓ Specify urgency indicators")
    print("   ✓ Set category hierarchies")
    print()

    print("5. Consistency")
    print("-" * 80)
    print("   ✓ Use consistent field names")
    print("   ✓ Maintain same JSON structure")
    print("   ✓ Align enrichment and context prompts")
    print()

    print("6. Testing")
    print("-" * 80)
    print("   ✓ Test with real messages from your domain")
    print("   ✓ Validate JSON structure")
    print("   ✓ Check importance scoring accuracy")
    print()


def main():
    """Run all examples."""

    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  Mindcore Custom Prompts - Complete Examples".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")

    example_1_view_default_prompts()
    input("Press Enter to continue...")

    example_2_formatted_prompts()
    input("Press Enter to continue...")

    example_3_custom_prompts_for_ecommerce()
    input("Press Enter to continue...")

    example_4_custom_prompts_yaml()
    input("Press Enter to continue...")

    example_5_specialized_prompts()
    input("Press Enter to continue...")

    example_6_multilingual_prompts()
    input("Press Enter to continue...")

    example_7_prompt_best_practices()

    print()
    print("=" * 80)
    print("All examples complete! 🎉")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Identify your domain-specific requirements")
    print("  2. Create custom prompts for your use case")
    print("  3. Save them to custom_prompts.yaml")
    print("  4. Configure Mindcore to use your prompts")
    print("  5. Test and iterate!")
    print()
    print("Resources:")
    print("  • Default prompts: mindcore/prompts.py")
    print("  • Configuration: mindcore/config.yaml")
    print("  • Documentation: README.md (Custom Prompts section)")
    print()


if __name__ == "__main__":
    main()
