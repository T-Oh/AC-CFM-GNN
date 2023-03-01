if false
    ls_tot=0;
    total_load_before=0;
    for i=1:length(clusterresult(3).bus(:,4))
            P1 = clusterresult(3).bus(i,3);
            Q1 = clusterresult(3).bus(i,4);
            P2 = clusterresult(4).bus(i,3);
            Q2 = clusterresult(4).bus(i,4);
            total_load_before = total_load_before + sqrt(P1^2 + Q1^2);
        if ((P1 ~= P2) || (Q1 ~= Q2))
            ls_tot = ls_tot + abs(sqrt(P1^2 + Q1^2) - sqrt(P2^2 + Q2^2));
            %disp("BUS: " + i)
            %disp([clusterresult(3).bus(i,3) , clusterresult(3).bus(i,4)])
            %disp([clusterresult(4).bus(i,3), clusterresult(4).bus(i,4)])
        end
    end
    disp(total_load_before)
    disp(ls_tot)
    disp(ls_tot/total_load_before)
end

if false
    for i=1:length(clusterresult(1).branch(:,1))
        if clusterresult(1).branch(i,1) == 3075
            if clusterresult(1).branch(i,2) == 3138
                disp(clusterresult(1).branch(i,11))
            end
        end
        if clusterresult(1).branch(i,1) == 3138
            if clusterresult(1).branch(i,2) == 3075
                disp(clusterresult(1).branch(i,11))
            end
        end
    end
end

if false
    for i=1:length(clusterresult(1).branch_tripped(:,1))
        if clusterresult(1).branch_tripped(i,1) ~= 0
            print(i)
        end
    end
end


for i=1:length(clusterresult(1).branch(:,14))
    if isnan(clusterresult(1).branch(i,14))
        disp(clusterresult(1).branch(i,11))
    end
end


        
