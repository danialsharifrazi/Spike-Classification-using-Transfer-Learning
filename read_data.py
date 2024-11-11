def read_each_class(dpi):
    import numpy as np
    import glob

    # main path
    main_dictionary=f'./Aedes aegypti primary neurons/Spontaneous_{dpi} dpi'

    # Load Control data
    addresses_ct=glob.glob(main_dictionary+'/Control/'+'*.txt')
    dataset_ct=np.zeros((1,61))
    for item in addresses_ct:
        dataset_part=np.loadtxt(item)
        dataset_ct=np.concatenate((dataset_ct,dataset_part),axis=0)
    dataset_ct=dataset_ct[1:,:]

    # Load DENV2 data
    addresses_dv=glob.glob(main_dictionary+'/DENV2 infected/'+'*.txt')
    dataset_dv=np.zeros((1,61))
    for item in addresses_dv:
        dataset_part=np.loadtxt(item)
        dataset_dv=np.concatenate((dataset_dv,dataset_part),axis=0)
    dataset_dv=dataset_dv[1:,:]

    # Load ZIKV data
    addresses_zk=glob.glob(main_dictionary+'/ZIKV infected/'+'*.txt')
    dataset_zk=np.zeros((1,61))
    for item in addresses_zk:
        dataset_part=np.loadtxt(item)
        dataset_zk=np.concatenate((dataset_zk,dataset_part),axis=0)
    dataset_zk=dataset_zk[1:,:]


    # create similar amount of data
    len1=dataset_ct.shape[0]
    len2=dataset_dv.shape[0]
    len3=dataset_zk.shape[0]
    len_min=min(len1,len2,len3)
    dataset_ct=dataset_ct[:len_min,1:]
    dataset_dv=dataset_dv[:len_min,1:]
    dataset_zk=dataset_zk[:len_min,1:]

    # show data shape seperately
    print('Control: ',dataset_ct.shape)
    print('Denv2: ',dataset_dv.shape)
    print('Zika: ',dataset_zk.shape)

    return dataset_ct, dataset_dv, dataset_zk

