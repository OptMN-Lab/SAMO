mkdir -p ./save
mkdir -p ./trainlogs

method=fairgradwithsamo
rho=0.00001
beta=0.9
zo_eps=0.01
seed=0

nohup python -u trainer_samo.py --method=$method --rho=$rho --seed=$seed --beta=$beta --zo_eps=$zo_eps --scale-y=True > trainlogs/$method-rho$rho-beta$beta-sd$seed-v2.log 2>&1 &
