instance_weight = 0

def update_instance_weight(sn_epoch):
    thre_begin = 0.1
    thre_end = 0.01

    if sn_epoch < 50:
        instance_weight = thre_begin
    elif sn_epoch < 250:
        instance_weight = thre_begin + (thre_end - thre_begin) * (sn_epoch - 50) / 200
    else:
        assert sn_epoch <= 400
        instance_weight = thre_end