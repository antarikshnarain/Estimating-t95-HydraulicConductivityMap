/*---------------------------------
 * Author: Antariksh Narain
 * Description: Main program
 *---------------------------------*/

#include "MapGenerator.cpp"
#include "ShortestDistance.cpp"
// #include <stdio.h>
// #include "Algorithm.cpp"


int main()
{
    MapGenerator mg;
    mg.ReadFile("map_test.csv");
    //mg.ReadFiles(5, new string[5]{"Field_Serra_2060_0.csv", "Field_Serra_2060_1.csv","Field_Serra_2060_2.csv", "Field_Serra_2060_3.csv", "Field_Serra_2060_4.csv"});
    ShortestDistance sd(mg.conductivity_map[0]);
    vector<NodeInfo> result = sd.ProbilisticPath(Pos(0,0));
    float total  = 0;
    for(int i=0;i<result.size();i++)
    {
        printf("Pos(%d, %d) : %f\n",result[i].idx.x, result[i].idx.y, result[i].conductivity);
        total += result[i].conductivity;
    }
    printf("Total: %f\n", total);
    return 0;
}