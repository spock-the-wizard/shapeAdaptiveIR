
exp_name="test/var14"

for name in cone4 pyramid4 cylinder4 custom
do
    echo Running $name
    echo $exp_name
    bash ./template/learn_$name.sh $exp_name
done