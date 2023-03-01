# (c) 2023 - Brian Jay Tang, University of Michigan, <bjaytang@umich.edu>
#
# This file is part of Eyeshield
#
# Released under the GPL License, see included LICENSE file

python eval/write_labels.py
python eval/new_process_cloud_results.py
python eval/parse_performance.py
python user_study/process_survey.py
python user_study/process_inperson_survey.py
