# Actor pretrain script created by wang
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

#newton cotes
def intg(datapack, T):
    assert datapack.shape[0] == 7, "data length for int was incorrect!"
    Ck = np.array([41, 216, 27, 272, 27, 216, 41],np.double) / 840.0
    itv = np.matmul(np.expand_dims(Ck, 1).T, datapack)
    return np.squeeze(itv)*T

def Actor_Features(x):
    x1, x2, x3, x4 = np.expand_dims(x[:, 0], 1), np.expand_dims(x[:, 1], 1), np.expand_dims(x[:, 2], 1), np.expand_dims(x[:, 3], 1)
    return np.hstack((x1,x2,x3,x4))
    #return np.hstack((x1, x2, np.cos(x1)*(x2*x2*x2), x2*(np.cos(x1)*np.cos(x1)), x2*np.cos(x1)))

if __name__ == '__main__':
    DIV1 = np.pi / 2048
    DIV2 = np.pi / 520
    DIV_U = 50
    # read data
    data = loadmat('save_data_01.mat')

    theta = data['THETA'] * DIV1
    pose = data['POSE'] * DIV2
    d_theta = data['D_THETA'] * DIV1 / 5
    d_pose = data['D_POSE'] * DIV2

    x_data = np.hstack((theta,d_theta, pose, d_pose))
    y_data = data['U'] / DIV_U
    uj = data['Uj'] / DIV_U

    t = np.linspace(0, 499*0.025, 800)

    actor_features = Actor_Features(x_data)

    R=1
    u = y_data
    du = actor_features * R * (u - uj)
    X = actor_features * (u - uj)#np.hstack((x_data, actor_features))*(u - uj)
    Gx = np.tanh(x_data)
    d_adv = np.array([np.kron(X[c], Gx[c]) for c in range(y_data.shape[0])])
    r = np.expand_dims(np.sum(np.multiply(x_data, x_data), 1), 1)*6 + np.square(uj) * R

    i = 0
    T = 0.025*6 # send a state every 25ms
    activate_u = []
    int_r = []
    int_adv = []
    TD = []
    r_diff = []
    while(i+6 < 800):
        td = np.kron(x_data[i+6], x_data[i+6]) - np.kron(x_data[i], x_data[i])
        TD.append(td)
        int_adv.append(intg(d_adv[i:i+7], T))
        activate_u.append(intg(du[i:i+7], T))
        int_r.append(intg(r[i:i+7], T))
        r_diff.append(r[i+6] - r[i])
        i+=1

    activate_u = np.array(activate_u)
    TD = np.array(TD)
    int_adv = np.array(int_adv)
    G = 5  #lambda
    H = np.hstack((TD, 2 * G * activate_u, int_adv))
    Y = G * np.expand_dims(int_r, 1) + np.array(r_diff)
    #H = np.hstack((TD, 2 * activate_u))
    #Y = np.expand_dims(int_r, 1)

    h_std = np.expand_dims(np.std(H, 0), 1).T
    H = H / h_std
    y_std = np.std(Y)
    Y = Y / y_std

    result = np.matmul(np.linalg.pinv(H), -Y)
    err = np.sqrt(np.mean(np.square(H.dot(result)+Y)))
    actor_weights = result[16:20] / h_std.T[16:20] * y_std * [[DIV1 * DIV_U],[DIV1 * DIV_U],[DIV2 * DIV_U],[DIV2 * DIV_U]] # * 50 / 7200 * 12 #Tranform unit.
    print('Err: ', err, 'ac_weights: ', actor_weights.T)
    #savemat('optimal.mat', {'critic': result[:16], 'actor':actor_weights})
    #np.save('actor.npy',result[16:20])


    af_dim = actor_features.shape[1]
    opt = np.matmul(x_data, actor_weights)
    plt.ioff()
    fig2 = plt.figure()
    ax = fig2.add_subplot(2,1,1)
    ax.plot(t, u)
    ax = fig2.add_subplot(2,1,2)
    ax.plot(t, np.squeeze(opt))
    plt.show()

# np.savetxt('W1.txt',W1,fmt='%.6f', newline='\n')
# np.savetxt('B1.txt',b1,fmt='%.6f', newline='\n')
# with open('feature_extractor.txt', 'w') as f:
#     W1.tofile(f, ', ')
#     f.write('\r\n')
#     b1.tofile(f, ', ')
