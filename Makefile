.PHONY: build verify index ask smoke ui shell

build:
	docker compose build

verify:
	docker compose run --rm rag python scripts/verify_install.py

index:
	docker compose run --rm rag python main.py index data/samples/ --fresh -e naive -e hybrid

ask:
	docker compose run --rm rag python main.py ask "What are the main components of a RAG system?" -e naive

smoke:
	docker compose run --rm rag python main.py run-all data/questions_smoke.txt data/samples/ -e naive -e hybrid -o results/smoke.json

ui:
	docker compose up streamlit

shell:
	docker compose run --rm cli
