

def brute_force_trainner(data_file, label_feature_name, min_features, step, verbose):
    """

    use rfe and cross validation to brute force the optimal number of features
    to use for a good LDA

    design to be very atonomous, provide the data_file and the name of
    the label variable within it

    min features is the minimum feature to keep (for stoping RFE)

    step is the number of feature to drop by iteration

    """

    # grid search solver for lda
    from sklearn.datasets import make_classification
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import pandas as pd
    from joblib import dump
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR

    ## load data
    X_data = pd.read_csv(data_file)

    ## hunt the different cluster name in the label feature
    cluster_name = []
    for label in list(X_data[label_feature_name]):
        if(label not in cluster_name):
            cluster_name.append(label)

    ## display extracted labels
    if(verbose):
        print("[+] => "+str(len(cluster_name))+" class detected")
        for cluster in cluster_name:
            print("[+]    - "+str(cluster))

    ## parse dataset
    X = X_data[X_data[label_feature_name].isin(cluster_name)]
    Y = X[label_feature_name]
    X = X.drop(columns=[label_feature_name])
    feature_list = list(X.keys())

    ## process label
    cmpt = 0
    label_to_encode = {}
    for label in cluster_name:
        label_to_encode[label] = cmpt
        cmpt +=1
    Y = Y.replace(label_to_encode)
    y = Y.values
    X = X.values

    ## define number_feature_to_select
    number_feature_to_select = len(feature_list)-step

    ## init log file
    log_file = open("picker_brute_force.log", "w")
    log_file.write("NB_FEATURES,SOLVER,ACC\n")

    ## RFE LOOP
    while(number_feature_to_select > min_features):

        ## run RFE
        if(verbose):
            print("[+] Running RFE => target "+str(number_feature_to_select)+" variables")
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=number_feature_to_select, step=1)
        selector = selector.fit(X, y)

        i = 0
        selected_features = []
        for keep in selector.support_:
            if(keep):
                selected_features.append(feature_list[i])
            i+=1
        selected_features.append(label_feature_name)

        #-> recraft dataset
        if(verbose):
            print("[+] Recraft dataset")
        X = X_data[X_data[label_feature_name].isin(cluster_name)]
        X = X[selected_features]
        Y = X[label_feature_name]
        X = X.drop(columns=[label_feature_name])
        feature_list = list(X.keys())
        Y = Y.replace(label_to_encode)
        y = Y.values
        X = X.values

        #-> update nb of features
        number_feature_to_select = i-step

        ## LDA Training
        # define model
        model = LinearDiscriminantAnalysis()

        # define model evaluation method
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1)

        # define grid
        grid = dict()
        grid['solver'] = ['svd', 'lsqr', 'eigen']

        # define search
        search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)

        # perform the search
        if(verbose):
            print("[+] Training LDA")
        results = search.fit(X, y)

        # summarize
        best_solver = results.best_params_['solver']
        best_score = results.best_score_

        if(verbose):
            print('[+]     -> Mean Accuracy: %.3f' % results.best_score_)
            print('[+]     -> Solver: %s' % results.best_params_['solver'])

        ## save results
        log_file.write(str(len(selected_features))+","+str(results.best_params_['solver'])+","+str(results.best_score_)+"\n")

        ## save features
        feature_file = open("features/rfe_determined_features_i"+str(i)+".csv", "w")
        feature_file.write("FEATURE\n")
        for f in feature_list:
            feature_file.write(str(f)+"\n")
        feature_file.close()

    ## close result file
    log_file.close()

def plot_acc(log_file):
    """
    Plot results of the LDA Exploration using content of log file
    """

    ## importation
    import pandas as pd
    import matplotlib.pyplot as plt

    ## load log file
    df = pd.read_csv(log_file)

    ## get the different solver
    x = df['NB_FEATURES']
    y = df['ACC']

    ## create plot
    plt.plot(x,y, '--bo')
    plt.title("Picker Exploration")
    plt.ylabel("ACC")
    plt.xlabel("Nb Features")
    plt.savefig("images/picker_exploration.png")
    plt.close()


def hunt_best_conf(log_file):
    """
    """

    ## importation
    import pandas as pd

    ## parameters
    max_acc = 0
    best_config = "NA"
    best_var_nb = "NA"

    ## load dataset
    df = pd.read_csv(log_file)

    ## parse data
    for index, row in df.iterrows():

        #-> extract data
        acc = row['ACC']
        solver = row['SOLVER']
        nb_features = row['NB_FEATURES']

        #-> test if acc is the best
        if(float(acc) > max_acc):
            max_acc = float(acc)
            best_config = solver
            best_var_nb = nb_features

    ## craft output data
    hunt_results = {
        "acc":max_acc,
        "solver":best_config,
        "features_number":best_var_nb
    }

    ## return data
    return hunt_results


def make_yourself_a_home():
    """
    create needed directory of they don't exist
    """

    ## importation
    import os

    ## parameters
    target_dir = ["features", "images"]
    for dir in target_dir:
        if(not os.path.isdir(dir)):
            os.mkdir(dir)


def display_help():
    """
    Display help for the programm
    """

    ## importation
    from colorama import init
    init(strip=not sys.stdout.isatty())
    from termcolor import cprint
    from pyfiglet import figlet_format

    ## display cool banner
    text="Picker - Help"
    cprint(figlet_format(text, font="standard"), "blue")

    ## help
    print("""
        Need 4 arguments:
            -i : input file (csv)
            -l : name of the label column in the input file
            -m : min number of features to keep (for RFE usage)
            -s : step for RFE algorithm, number of features to drop at each iteration

        Can handle 1 optional argument:
            -v : verbose, optional arguments, set to True by default

        Exemple of use:
            python picker.py -i dummy_test_data.csv -l LABEL -m 10 -s 5
    """)


def run(input_data_file, label_name, min_nb_features, step, verbose):
    """
    """

    ## make yourself confortable
    make_yourself_a_home()

    ## parameters
    log_file = "picker_brute_force.log"

    ## brute force LDA process
    brute_force_trainner(
        input_data_file,
        label_name,
        min_nb_features,
        step,
        verbose
    )

    ## generate image from log file
    plot_acc(log_file)

    ## hunt best configuration
    results = hunt_best_conf(log_file)

    ## display best configuration
    print("-"*45)
    print("[*] Best ACC => "+str(results['acc']))
    print("[*] With solver => "+str(results['solver']))
    print("[*] With Optimal nb features => "+str(results['features_number']))


##------##
## MAIN ########################################################################
##------##
if __name__=='__main__':

    ## importation
    import sys
    import getopt
    from colorama import init
    init(strip=not sys.stdout.isatty())
    from termcolor import cprint
    from pyfiglet import figlet_format

    ## catch arguments
    argv = sys.argv[1:]

    ## parse arguments
    input_data_file = ''
    label_name = ''
    min_nb_features = ''
    step = ''
    verbose = ''
    try:
       opts, args = getopt.getopt(argv,"hi:l:m:s:v:",["ifile=","label=","min=","step=","verbose="])
    except getopt.GetoptError:
       display_help()
       sys.exit(2)
    for opt, arg in opts:
       if opt in ('-h', '--help'):
           display_help()
           sys.exit()
       elif opt in ("-i", "--ifile"):
           input_data_file = arg
       elif opt in ("-l", "--label"):
           label_name = arg
       elif opt in ("-m", "--min"):
           min_nb_features = int(arg)
       elif opt in ("-s", "--step"):
           step = int(arg)
       elif opt in ("-v", "--verbose"):
           verbose = bool(arg)

    ## display cool banner
    text="Picker - Dimension Reduction"
    cprint(figlet_format(text, font="standard"), "blue")

    ## check that all arguments are present
    if(input_data_file == ''):
        print("[!] No input file detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(label_name == ''):
        print("[!] No label detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(min_nb_features == ''):
        print("[!] No min_nb_features detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(step ==''):
        print("[!] No step detected")
        print("[!] Use -h or --help options to get more informations")
        sys.exit()
    if(verbose == ''):
        verbose = True

    ## perform run
    run(input_data_file, label_name, min_nb_features, step, verbose)
