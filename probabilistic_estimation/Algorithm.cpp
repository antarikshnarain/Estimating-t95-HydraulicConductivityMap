/*---------------------------------
 * Author: Antariksh Narain
 * Description: Our Algorithm
 *---------------------------------*/

#include "MapGenerator.cpp"
#include "ShortestDistance.cpp"

int main()
{
    MapGenerator mg;
    // int total_maps = mg.ReadFile("map_test.csv");
    int total_maps = mg.ReadFiles(5, new string[5]{"Field_Serra_2060_0.csv", "Field_Serra_2060_1.csv","Field_Serra_2060_2.csv", "Field_Serra_2060_3.csv", "Field_Serra_2060_4.csv"});
    // print header
    printf("Total Maps, %d\n", total_maps);
    printf("Seq ");
    for(int i=0;i<MAP_ROWS;i++)
    {
        printf(", Pos(%d:0)", i);
    }
    printf("\n");
    // for(int i=0;i<total_maps;i++)
    // {
    //     printf("%d ", i);
    //     ShortestDistance sd(mg.conductivity_map[i]);
    //     for(int r=0;r<MAP_ROWS;r++)
    //     {
    //         vector<NodeInfo> result = sd.ProbilisticPath(Pos(r,0));
    //         float total  = 0;
    //         for(int i=0;i<result.size();i++)
    //         {
    //             //printf("Pos(%d, %d) : %f\n",result[i].idx.x, result[i].idx.y, result[i].conductivity);
    //             total += result[i].conductivity;
    //         }
    //         printf(", %f", total);
    //     }
    //     printf("\n");
    // }
    
    mg.DumpGrid(2);
    ShortestDistance sd(mg.conductivity_map[2]);
    sd.DumpPath(sd.ProbilisticPathAll());
}