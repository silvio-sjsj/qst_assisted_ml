% M = 8 or 14

% Pauli matrices
sx = sparse([0 1; 1  0]);
sz = sparse([1 0; 0 -1]);
id = speye(2);

% operators acting on spins
sxns = cell(M,1);
szns = cell(M,1);
for iM = 1:M
    sxns{iM} = kron(kron(speye(2^(M-iM)),sx) , speye(2^(iM-1)));
    szns{iM} = kron(kron(speye(2^(M-iM)),sz) , speye(2^(iM-1)));
end

% Hamiltonian
H = sparse(2^M,2^M);
for i1 = 1:M
    for i2 = 1:M
        if i1~=i2
            H = H + Jij(i1,i2) * sxns{i1}*sxns{i2};
        end
    end
    H = H + (B+Bi(i1)) * szns{i1};
end

timeEvol = expm(-1i*dt*H);

% or use the additional function expv instead of expm for large M