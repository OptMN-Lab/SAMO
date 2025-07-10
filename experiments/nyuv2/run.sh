mkdir -p ./save
mkdir -p ./trainlogs

method=fairgradwithsamo
seed=0
rho=0.001
beta=0.5

CUDA_VISIBLE_DEVICES=0 nohup python -u trainer_samo.py --method=$method --rho=$rho --beta=$beta --seed=$seed > trainlogs/$method-rho$rho-beta$beta-sd$seed.log 2>&1 &
