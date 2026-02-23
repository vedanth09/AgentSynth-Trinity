# AgentSynth-Trinity Project Makefile

.PHONY: install run test audit-report clean

install:
	pip install -r requirements.txt
	pip install pydantic==2.0.0 ml-flow shap matplotlib scipy xgboost

run:
	streamlit run main.py

test:
	pytest tests/

audit-report:
	@echo "Generating GDPR Article 30 Audit Report..."
	@cat audit_log.json | jq .

clean:
	rm -rf __pycache__ .pytest_cache
	rm -f trinity_radar.png benchmark_report.json compliance_certificate.txt
