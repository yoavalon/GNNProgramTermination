# GNNProgramTermination
Using Graph Neural Networks for Program Termination

If not installed, install python virtual environment : 
> pip install virtualenv 

#create virtual environment with directory
python3 -m venv gnnEnv

#activate virtual environment
source gnnEnv/bin/activate

#show all installed packages
pip list

#extract requirement file 
pip freeze > requirements.txt

#in environment load the requirements file 
python3 -m pip install -r requirements.txt
