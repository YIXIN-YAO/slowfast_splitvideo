from split import vis

csv_path = './csv_res/50.csv'
label_list = list()
with open(csv_path, "r") as f:
    for idx, line in enumerate(f.read().splitlines()):
        assert (len(line.split(' ')) == 3)
        task_id, label, score = line.split(' ')
        label_list.append(int(label))
    vis(label_list)
