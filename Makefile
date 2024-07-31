.PHONY: install-dev clean format check install uninstall test diff-test


format:
	black classifier offlinerl utils 
	blackdoc classifier offlinerl utils 
	isort classifier offlinerl utils 

check:
	black classifier offlinerl utils --check --diff
	blackdoc classifier offlinerl utils --check
	flake8 --config pyproject.toml --ignore E203,E501,W503,E741 classifier offlinerl utils
	mypy --config pyproject.toml classifier offlinerl utils
	isort classifier offlinerl utils --check --diff

pull:
	git pull origin main