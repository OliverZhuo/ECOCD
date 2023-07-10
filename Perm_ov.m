function P = Perm_ov(F, A)


Perm_ov = 0.00000;

D = sum(A,2); % n x 1

[n,k] = size(F);

for c=1:k

    for i=1:n

        if F(i,c) == 1
            
            % E
            E_max = 0;
            for com=1:k
                if com ~= c
                    E = A(i,:)*F(:,com);
                    if E > E_max
                        E_max = E;
                    end
                end
            end


            %I
            community = F(:, c);
            I = A(i, :) * community;
            in_nodes = find(community == 1);
            subgraph = A(in_nodes, in_nodes);

            % c_in
            if sum(community) < 3 || sum(subgraph(:)) < 6
                c_in = 0;
            else
                in_neighbours = A(:, i) .* community;
                in_neighb_ind = find(in_neighbours == 1);
                subgraph2 = A(in_neighb_ind, in_neighb_ind);
    
                numer = sum(subgraph2(:)) ./ 2;
                denom = 0.5 .* I .* (I - 1);
            
                if denom == 0
                    c_in = 0;
                else
                    c_in = numer ./ denom;
                end
            end %if sum(com) < 3 || sum(subgraph(:)) < 6
            

            if E_max == 0
                E_max = 1;
            end
            P = (I) / (E_max * D(i)) - (1 - c_in);
            Perm_ov = Perm_ov + P;
            

        end % F(i,c)
        

    end % for node
end % for community
    disp("Perm_ov");
    disp(Perm_ov);
Perm_ov = Perm_ov /sum(sum(F));
end