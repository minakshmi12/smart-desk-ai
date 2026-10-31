# scripts/langsmith_check.py  (replace)
"""Show where the datasets and experiments actually live."""
import os

from dotenv import load_dotenv

load_dotenv()

from langsmith import Client  # noqa: E402

client = Client()
print("Endpoint:", client.api_url)
print("Project env:", os.getenv("LANGSMITH_PROJECT"))

for d in client.list_datasets():
    if not d.name.startswith("smartdesk-qa"):
        continue
    print("\nDATASET:", d.name)
    print("  id:", d.id)
    print("  examples:", getattr(d, "example_count", "?"))
    print("  url:", getattr(d, "url", "(not provided by this SDK version)"))
    exps = list(client.list_projects(reference_dataset_id=d.id))
    print("  experiments:", len(exps))
    for e in exps:
        print("   -", e.name)