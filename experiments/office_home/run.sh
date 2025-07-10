mkdir -p ./save
mkdir -p ./trainlogs

method=fairgradwithsamo
rho=0.001
beta=0.5
seed=0

CUDA_VISIBLE_DEVICES=0 nohup python -u trainer_samo.py --method=$method --rho=$rho --beta=$beta --seed=$seed > trainlogs/$method-rho$rho-beta$beta-sd$seed.log 2>&1 &
