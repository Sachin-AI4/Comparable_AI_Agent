# import json
# from src.agent.graph import create_agent_graph

# def main():
#     """Main Entry point for domain comparable search."""
#     app = create_agent_graph()

#     print("\n" + "="*80)
#     print("DOMAIN COMPARABLE SEARCH AGENT")
#     print("="*80)
#     input_domain = input("\n Enter domain name : ").strip()

#     if not input_domain:
#         print("Error: Domain name cannot be empty!")

    

#     initial_state = {
#         "input_domain": input_domain
#     }

#     result = app.invoke(initial_state)

#     # print result
#     if result.get("error"):
#         print(f"Error: {result['error']}")
    
#     else:
#         output = result.get("result", {})
        
#         # Print JSON output
#         print("\n" + "="*80)
#         print("FINAL JSON OUTPUT")
#         print("="*80)
#         print(json.dumps(output, indent=2))

#         # Summary
#         print(f"\n{'='*80}")
#         print(f"SUMMARY")
#         print(f"{'='*80}")
#         print(f"Domain: {output['input_domain']}")
#         print(f"Categories: {output['primary_category']}, {output['secondary_category']}")
#         print(f"Found {output['total_comparables']} comparables (confidence: {output['confidence']})")
#         print(f"{'='*80}\n")

#         print("COMPARABLE DOMAINS:")
#         for i, comp in enumerate(output['comparables'], 1):
#             print(f"{i}. {comp['domain']}")
#             print(f"   Price: ${comp['price']:,.0f}")
#             print(f"   Date: {comp['date']}")
#             print(f"   Score: {comp['score']:.3f} (semantic: {comp['semantic_sim']:.3f})")
#             print()

# if __name__ == "__main__":
#     main()