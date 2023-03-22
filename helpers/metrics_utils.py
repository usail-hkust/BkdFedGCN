import os

def log_test_results(model_path, list, file_name):
    'Given list, transfer list to string, and write is to csv'
    string = ','.join(str(n) for n in list)
    path = model_path + '/test_results'
    if not os.path.isdir(path):
        os.makedirs(path + '/', )

    file_path = path + "/{}.csv".format(file_name)

    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)