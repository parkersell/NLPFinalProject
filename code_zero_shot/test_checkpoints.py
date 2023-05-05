from subprocess import Popen


for i in range(5, 50, 5):
    processes = []
    for j in ['ing', 'verb', 'sent']:
        processes.append(Popen(['python3', 'inference_videosGT.py', '--ckpt_num', f'-{i}', '--eval_type', j]))
    for p in processes:
        p.wait()



   