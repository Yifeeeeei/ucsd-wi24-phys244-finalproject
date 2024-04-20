models: models.c
	clang models.c -o models -I/opt/homebrew/opt/openblas/include -L/opt/homebrew/opt/openblas/lib -lopenblas  