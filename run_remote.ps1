scp -r C:\Users\christian\Documents\git_files\CompArchCompetition ex2023@comparc01.am.ics.keio.ac.jp:~/
 
ssh -t ex2023@comparc01.am.ics.keio.ac.jp "cd ~/CompArchCompetition ;export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH; make clean; make; ./conv"
