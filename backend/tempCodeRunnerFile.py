z = 0
for k in weight_0:
    ctr = 0
    mark = 0
    for i in w_base:
        if ctr == 0:
            euclidian_distance = np.linalg.norm(k-i)
            cos_sim = cosine_sim(k,i)
        else:
            if np.linalg.norm(k-i) < euclidian_distance:
                euclidian_distance = np.linalg.norm(k-i)
                mark = ctr
                cos_sim = cosine_sim(k,i)
        ctr += 1
    print(f"yang sesuai dengan dataset: {label_training[mark]}")
    print(f"label test {z}: {label_test[z]}")
    print(f"cosine similiarity: {cos_sim}")
    z+= 1
