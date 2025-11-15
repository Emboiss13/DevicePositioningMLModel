clc; clear;

roomSize = [10 10];

% Test 1: Create one random wall
disp("Creating a random wall...");
w = Wall.randomWall(roomSize);

disp("Wall created:");
disp(w);

% Display details
fprintf("  p1 = [%.2f %.2f]\n", w.p1(1), w.p1(2));
fprintf("  p2 = [%.2f %.2f]\n", w.p2(1), w.p2(2));
fprintf("  width = %.2f\n", w.width);
fprintf("  material = %s\n", w.materialName);
fprintf("  attenuation = %.2f dB\n", w.attenuation);

% Test 2: Create multiple walls
disp("Creating 5 random walls...");
walls = Wall.empty;

for i = 1:5
    walls(i) = Wall.randomWall(roomSize);
end

disp("Walls created:");
for i = 1:5
    fprintf("Wall %d: material= %s, att= %.1f || ", ...
        i, walls(i).materialName, walls(i).attenuation);
end
