
exp_name="exp1/var41"
spp=256

for name in buddha1 kettle1 cone4 sphere1 head1 pyramid4 duck cylinder4 
do
    echo Running $name
    echo $exp_name
    bash ./template/learn_$name.sh $exp_name $spp
done