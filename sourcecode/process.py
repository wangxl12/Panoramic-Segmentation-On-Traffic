from multiprocessing import Process
import multiprocessing as mp

try:
    mp.set_start_method('spawn')
except:
    pass


def process1(input_path, output_path, time_interval):
    print('start processing 1...')
    with ProcessVideo(input_path, output_path, time_interval=time_interval) as p1:
        p1.run()


def process2(input_path, output_path, time_interval):
    print('start processing 2...')
    with ProcessVideo(input_path, output_path, time_interval=time_interval) as p2:
        p2.run()
        
if __name__ == '__main__':
    processes = [process1, process2]
    jobs = []
    for pro in processes:
        p = Process(target=pro, args=(input_path, output_path, args.time_interval, ))
        jobs.append(p)
        p.start()
    for job in jobs:
        job.join()