# Define variables
ENV_NAME = scgpt_yml
YAML_FILE = conda.yaml

# Setup the conda environment and install flash-attn
setup:
	@echo "Creating conda environment from $(YAML_FILE)..."
	conda env create -f $(YAML_FILE)
	@echo "Activating environment and installing flash-attn..."
	conda run -n $(ENV_NAME) pip install flash-attn==1.0.4 --no-build-isolation
	@echo "Environment setup complete."
	@echo "Registering ipykernel..."
	conda run -n $(ENV_NAME) python -m ipykernel install --user --name=$(ENV_NAME)
	@echo "ipykernel registered."

# Remove the conda environment
clean:
	@echo "Removing conda environment $(ENV_NAME)..."
	conda env remove -n $(ENV_NAME)
	@echo "Environment removed."

# Activate the environment (alias for convenience)
activate:
	@echo "To activate the environment, run: conda activate $(ENV_NAME)"

# List available targets
help:
	@echo "Available targets:"
	@echo "  setup      - Create the conda environment and install post-build dependencies"
	@echo "  clean      - Remove the conda environment"
	@echo "  activate   - Command to activate the environment"
	@echo "  help       - List available targets"
