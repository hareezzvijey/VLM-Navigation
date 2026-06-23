"""
VLM Pipeline - Entry Point with Grok/Phi-2 LLM Support
"""
import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import VLMPipeline
from config.paths import OUTPUTS_DIR
from llm.client import LLMClient


def main():
    parser = argparse.ArgumentParser(
        description="VLM Navigation Pipeline with Grok LLM Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rule-based only (fastest, no API)
  python main.py --image street.jpg

  # Grok (recommended - free credits available)
  python main.py --image street.jpg --llm --llm-provider grok --llm-model grok-4.3

  # Phi-2 via Ollama (local, free)
  python main.py --image street.jpg --llm --llm-provider ollama --llm-model phi:2.7b

  # With depth estimation
  python main.py --image street.jpg --depth --llm --llm-provider ollama

  # WalkGPT-style rich output
  python main.py --image street.jpg --walkgpt --llm --llm-provider ollama
        """
    )
    
    # ── Core arguments ──────────────────────────────────────────────────────
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--no-sam", action="store_true", help="Disable SAM segmentation")
    parser.add_argument("--max-size", type=int, default=800, help="Max image size for processing")
    parser.add_argument("--output", default="output.png", help="Output filename")
    
    # ── Depth flags ─────────────────────────────────────────────────────────
    parser.add_argument("--depth", action="store_true", help="Enable MiDaS monocular depth estimation")
    parser.add_argument("--depth-model", default="MiDaS_small", 
                        choices=["MiDaS_small", "DPT_Hybrid", "DPT_Large"], 
                        help="Which MiDaS model to use for depth estimation")
    
    # ── LLM flags ───────────────────────────────────────────────────────────
    parser.add_argument("--llm", action="store_true", help="Enable LLM-enhanced guidance generation")
    parser.add_argument("--llm-provider", default="grok", 
                        choices=["grok", "ollama", "transformers", "vllm", "openai", "anthropic", "gemini"],
                        help="LLM provider to use (grok recommended)")
    parser.add_argument("--llm-model", default="grok-4.3",
                        help="LLM model name (e.g., grok-4.3, phi:2.7b, gemini-2.5-flash)")
    parser.add_argument("--llm-api-key", help="API key for LLM provider (required for Grok/Gemini/OpenAI)")
    
    # ── Output mode ─────────────────────────────────────────────────────────
    parser.add_argument("--walkgpt", action="store_true", help="Use WalkGPT-style rich conversational output")
    
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # ── Create output directory ────────────────────────────────────────────
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    # ── Initialize LLM client if enabled ────────────────────────────────────
    llm_client = None
    if args.llm:
        print("\n" + "=" * 60)
        print("  INITIALIZING LLM")
        print("=" * 60)
        print(f"  Provider: {args.llm_provider}")
        print(f"  Model   : {args.llm_model}")
        
        llm_client = LLMClient(
            model=args.llm_model,
            provider=args.llm_provider,
            api_key=args.llm_api_key,
        )
        
        if llm_client.load():
            print("LLM loaded successfully")
        else:
            print("LLM failed to load. Falling back to rule-based.")
            llm_client = None
        print("=" * 60)

    # ── Run pipeline ──────────────────────────────────────────────────────
    pipeline = VLMPipeline(
        max_image_size=args.max_size,
        enable_depth=args.depth,
        depth_model_type=args.depth_model
    )
    
    results = pipeline.detect_and_segment(args.image, run_sam=not args.no_sam)
    
    # ── Generate description ──────────────────────────────────────────────
    if args.walkgpt:
        nav = pipeline.build_walkgpt_description(results)
        # WalkGPT doesn't use LLM directly, but we can add it
        if args.llm and llm_client:
            # FIX: Correct import path
            from llm.prompt import build_walkgpt_prompt
            nav_data = {
                **nav,
                "accessibility": nav.get("accessibility", {}),
                "spatial_map": nav.get("spatial_map", {}),
            }
            prompt = build_walkgpt_prompt(nav_data)
            llm_response = llm_client.generate(prompt)
            if llm_response:
                nav["guidance"] = llm_response
                nav["guidance_source"] = "llm"
    else:
        nav = pipeline.build_navigation_description(
            results,
            use_llm=args.llm and llm_client is not None,
            llm_client=llm_client
        )

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  NAVIGATION OUTPUT")
    print("=" * 60)
    
    if args.walkgpt:
        # WalkGPT rich output
        print(f"  ACTION: {nav['action']}")
        print(f"  RISK  : {nav['risk'].upper()}")
        print(f"  SOURCE: {nav.get('guidance_source', 'rule_based').upper()}")
        print("\n  [GUIDANCE]")
        print(f"  {nav.get('guidance', nav.get('scene_text', ''))}")
        print("\n  [SPATIAL MAP]")
        for d in ["left", "center", "right"]:
            sm = nav['spatial_map'][d]
            obj_str = ", ".join(sm['objects']) if sm['objects'] else "clear"
            print(f"  {d.capitalize():<6}: {sm['status']:<10} | {obj_str}")
        if nav.get("accessibility"):
            print("\n  [ACCESSIBILITY]")
            acc = nav['accessibility']
            print(f"  Surface: {acc.get('surface', 'unknown')}")
            if acc.get('features'):
                print(f"  Features: {', '.join(acc['features'])}")
            if acc.get('hazards'):
                print(f"  Hazards: {', '.join(acc['hazards'])}")
            print(f"  Width: {acc.get('width_assessment', 'unknown')}")
    else:
        # Original compact output
        print(f"  Action     : {nav['action']}")
        print(f"  Risk       : {nav['risk']}")
        print(f"  Free-space : {nav.get('free_space', {})}")
        if nav.get("obstacles"):
            print(f"  Obstacles  : {nav['obstacles']}")
        if nav.get("surfaces"):
            print(f"  Surfaces   : {nav['surfaces']}")
        print(f"  Source     : {nav.get('guidance_source', 'rule_based').upper()}")
        print(f"\n  LLM-ready  :\n{nav.get('scene_text', '')}")
        if nav.get("guidance"):
            print(f"\n  Natural Guidance:\n{nav['guidance']}")
        
    print("=" * 60)

    # ── Visualize ──────────────────────────────────────────────────────────
    pipeline.visualize(results, save_name=args.output, show_depth=args.depth)


if __name__ == "__main__":
    main()