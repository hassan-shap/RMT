

for Ws = 1:5
    LNs = [0];
    for j = 1:9

        dA1 = 2^j;
        dB1 = 2^1;
        dA2 = 2^(10-j);
        dB2 = 2^1;
        dW = 2^Ws;

        T1 = tenzeros([dA1, dB1,dW]);
        T2 = tenzeros([dA2, dB2,dW]);

        T1c = tenzeros([dA1, dB1,dW]);
        T2c = tenzeros([dA2, dB2,dW]);

        rand1 =randn(dA1, dB1,dW)+1i*randn(dA1, dB1,dW);
        rand2 = randn(dA2, dB2,dW)+1i*randn(dA2, dB2,dW);

        T1(:,:,:) =rand1;
        T2(:,:,:) =rand2;

        T1c(:,:,:) =conj(rand1);
        T2c(:,:,:) =conj(rand2);

        Ttot = contract(ttt(T1, T2),3,6);
        Ttotc = contract(ttt(T1c, T2c),3,6);

        rhoA = contract(contract(ttt(Ttot,Ttotc), 2,6),3,6);

        tracerho = contract(contract(rhoA,1,3),1,2);

        rhoA = rhoA./tracerho;

        rhoAT2 = tenmat(rhoA, [3,2],[1,4]).data;

        LNs = [LNs, log(sum(abs(eig(rhoAT2))))];
        disp(log(sum(abs(eig(rhoAT2)))))
    end
    LNs = [LNs, 0];
    plot(0:10,LNs)
    hold on
end
% disp(log(dW))
% 
% dA1eff = min(dA1, dB1*dW);
% dA2eff = min(dA2, dB2*dW);
% 
% dB1eff = min(dB1, dA1*dW);
% dB2eff = min(dB2, dA2*dW);
% 
% disp(log(dA1eff*dA2eff/(dB1eff*dB2eff))/2)







