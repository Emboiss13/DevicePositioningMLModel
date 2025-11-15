classdef Materials
    properties (Constant)
        % A list (array of structs) of all materials
        List = struct( ...
            'name',   {'Wood', 'Glass', 'Brick', 'Concrete', 'Metal', 'Foliage'}, ...
            'atten',  {   3   ,    2   ,   12  ,     18    ,   25   ,     1     } ...
        )
    end

    methods (Static)
        function m = random()
            idx = randi(numel(Materials.List));
            m = Materials.List(idx);  % returns a struct with name & atten
        end

        function atten = getAttenuationByName(name)
            list = Materials.List;
            idx = find(strcmp({list.name}, name), 1);
            if isempty(idx)
                error('Unknown material name: %s', name);
            end
            atten = list(idx).atten;
        end
    end
end
