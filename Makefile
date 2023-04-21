MODULE_NAME=inv_cooking

format:
	autoflake --in-place --remove-all-unused-imports --ignore-init-module-imports --recursive $(MODULE_NAME)
	autoflake --in-place --remove-unused-variables --recursive $(MODULE_NAME)
	autoflake --in-place --expand-star-imports --recursive $(MODULE_NAME)
	isort -rc $(MODULE_NAME)
	black $(MODULE_NAME)

test:
	pytest $(MODULE_NAME) -s

check:
	black --check $(MODULE_NAME)
