# $1: full path to python script


if [ $1 == "--help" ]
then
	echo "first arg is full path of python script"
	echo "additional arguments are passed to python script"
	echo "additional arguments are path to model file and"
	echo "name of model (pkl) file"
elif [ $# == 0 ]
then
	echo "missing full-path of python script"
elif [ $# == 1 ]
then
	echo "launching streamlit: " $1
	streamlit run $1
elif [ $# > 1 ]
then
	echo "launching streamlit: " $1  " -- " ${@:2}
	streamlit run $1 -- ${@:2}
fi
