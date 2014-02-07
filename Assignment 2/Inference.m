g = computeG();
U = computeU();
for i = 1:size(U,1)
    [z yhat(i)] = max(U(i:) + g(i:), [], 2);

    