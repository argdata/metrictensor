Data Science Toolkit
==============

About the repository
--------------

**Graduates from Rutgers University** 

*Creators*: 
Emmanuel Contreras-Campana, PhD <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Christian Contreras-Campana, PhD

- November 2017

This repository contains an example jupyter notebook. The dskit directory has 
the data science visualization, optimization, feature_selection, keras_model,
and utilities toolkits. You may store cvs files in the data directory. Lastly, 
the model persistent files are meant to be stored in the models directory.


Installation
--------------

The data science toolkit repository may be install using pip.
Use the following command:
```
pip install -e git+https://github.com/ecampana/dskit#egg=dskit --process-dependency-links
```

Development
--------------

For the purposes of code development the pip install version will
need to be removed. Use the following command:
```
pip uninstall dskit
```
Then a local copy of the git repository will need to be checked out.
These are the instructions on how to install and run local package:

1. Inside repository:
```
python setup.py install
```
2. Start Jupyter notebook or restart Jupyter notebook depending on your
circumstance.

3. Import the package using:
```
from dskit import *
```

4. Additional python libraries that may be required can be installed using:
```
pip install -r requirements.txt
```
