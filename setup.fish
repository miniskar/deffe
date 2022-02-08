set SCRIPT_FILE (status filename)
set SCRIPT (python -c "import os,sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))" $SCRIPT_FILE)
set -gx DEFFE_DIR (dirname -- $SCRIPT)
echo "*********************** Deffe Environent *********************"
echo "DEFFE_DIR: $DEFFE_DIR"
echo "*************************************************************"
