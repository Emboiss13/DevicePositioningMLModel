classdef Wall
    properties
        materialName  
        attenuation
        p1
        p2
        width
    end

    methods
        function obj = Wall(p1, p2, materialName, width)
            obj.p1 = p1;
            obj.p2 = p2;
            obj.width = width;
            obj.materialName = materialName;
            obj.attenuation = Materials.getAttenuationByName(materialName);
        end
    end

    methods (Static)
        function w = randomWall(roomSize)
            % Random endpoints
            p1 = rand(1,2) .* roomSize;
            p2 = rand(1,2) .* roomSize;

            % Random width
            width = 0.1 + rand * 0.3;

            % Random material from Materials class
            m = Materials.random();   % struct with fields: name, atten

            % Create wall
            w = Wall(p1, p2, m.name, width);
            % attenuation gets filled via constructor
        end
    end
end

