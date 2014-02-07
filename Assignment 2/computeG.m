function g = computeG(x, allY)
    g = zeroes(size(x), size(allY), size(allY));
    for i = 1:size(x)
        for j = 1:size(allY)
            for k = 1:size(allY)
                g(i, j, k) = gi(allY(j), allY(k), x, i);
            end
        end
    end