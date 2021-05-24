val = np.zeros((3077,1))  
train_var = Variable(train_tensor.float())
for i in range(0,200):
    net = Net_two(input_size, 400, num_classes)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    y_train_value = y_train_values.iloc[:,i]
    y_train_tensor = torch.tensor(y_train_value.values)
    y_train_tensor = y_train_tensor.reshape([6154,1])
    print(y_train_tensor.shape)
    Y = y_train_tensor.float()
    print("Classifier: ", i)
    prev_loss = 1
    for epoch in range(1,300):
            optimizer.zero_grad()                             
            outputs = net(train_var)                             
            loss = criterion(outputs, Y)
            if loss < 0.01:
                print ('number of epoch', epoch, 'loss', loss.data)
                break
            prev_loss = loss
            loss.backward()                                  
            optimizer.step() 
            if epoch % 10 == 0:
                print ('number of epoch', epoch, 'loss', loss.data)
    TEST = Variable(test_tensor.float())
    predict_out = net(TEST)
    predict_out = predict_out.detach().numpy()
    
    #Saving the percentages 
    val = np.append(val, predict_out, axis=1)


val = val[:, range(1,201)]
print(val)


sample = pd.read_csv('/Users/sydneyeddamatthys/csci5461/project/y_test_sample.csv', index_col=0) 
df = pd.DataFrame(data=val, index=sample.index,  columns=sample.columns)
df.to_csv('test_predictions_6.csv')