library(argparse)
library(evtree)
parser = ArgumentParser(description='evtree command line parsing')
parser$add_argument('file_path', type="character", help='dataset path')
parser$add_argument('depth', type='integer', help='maximum allowed depth')
parser$add_argument('reg', type='double', help='regularization')
parser$add_argument('-t', '--test', type="character", help='test dataset')

# parse arguments
args = parser$parse_args()

# data path
file_path = args$file_path
test_file_path = args$test

# load data frame
data = read.csv(file_path)
test <- FALSE
if (! is.null(test_file_path)){
  test_data <- read.csv(test_file_path)
  test <- TRUE
}

# control
max_depth = args$depth
reg = args$reg

target_col <- tail(names(data), n=1)
c <- evtree.control(minbucket = 1L, minsplit = 2L, maxdepth = max_depth, alpha = reg, seed = 666) # seeds to ensure reproduction
start_time <- proc.time()
# start training
t <- eval(parse(text = paste('evtree(', target_col, '~ ., data = data, control = c)', sep = '')))
duration <- proc.time() - start_time

num_leaves <- width(t)
target <- eval(parse(text = paste('data$', target_col, sep = '')))
mse <- mean((target - predict(t))^2)
cat("Train Loss:", mse)
cat("Number of Leaves:", num_leaves)
cat("Training Duration:", duration, 'seconds')
if (test){
    test_target <- eval(parse(text = paste('test_data$', target_col, sep = '')))
    test_mse <- mean((test_target - predict(t, test_data))^2)
    cat("Test Loss:", test_mse)
}


