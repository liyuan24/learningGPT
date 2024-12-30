notebook: the_gpt_dev.py
	jupytext --to ipynb gpt_dev.py

py: gpt_dev.ipynb
	jupytext --to py:percent gpt_dev.ipynb