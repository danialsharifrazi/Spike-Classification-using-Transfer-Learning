def results(lst_acc,lst_f1,lst_precision,lst_recall,lst_matrix,lst_times,model_name,dpi):    
    import numpy as np
    
    results_path=f'./results_{model_name}.txt'
    f1=open(results_path,'a')

    f1.write('\nAverage Accuracy: '+str(np.mean(lst_acc)))
    f1.write('\nAverage Precision: '+str(np.mean(lst_precision)))
    f1.write('\nAverage Recall: '+str(np.mean(lst_recall)))
    f1.write('\nAverage F1 Score: '+str(np.mean(lst_f1)))
    f1.write('\nAverage Training Time: '+str(np.mean(lst_times)))

    f1.write('\n\n\nMetrics for all Folds: \n')

    for i in range(len(lst_acc)):
        f1.write('\n Accuracy: '+str(lst_acc[i]))
        f1.write('\n Precision: '+str(lst_precision[i]))
        f1.write('\n Recall: '+str(lst_recall[i]))
        f1.write('\n F1 Score: '+str(lst_f1[i]))
        f1.write('\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i]))
        f1.write('\n\n____________________\n')
    f1.close()
    

