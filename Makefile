# Define variables
ENV_NAME = scgpt_yml
YAML_FILE = conda.yaml

setup:
	@echo "Creating conda environment from $(YAML_FILE)..."
	conda env create -f $(YAML_FILE)
	@echo "Activating environment and installing flash-attn..."
	conda run -n $(ENV_NAME) pip install flash-attn==1.0.4 --no-build-isolation
	@echo "Environment setup complete."
	@echo "Registering ipykernel..."
	conda run -n $(ENV_NAME) python -m ipykernel install --user --name=$(ENV_NAME)
	@echo "ipykernel registered."
	unzip data/replogle.zip -d data/

clean:
	@echo "Removing conda environment $(ENV_NAME)..."
	conda env remove -n $(ENV_NAME)
	@echo "Environment removed."
