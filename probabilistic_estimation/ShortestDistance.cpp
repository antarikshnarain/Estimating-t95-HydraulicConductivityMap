/*---------------------------------
 * Author: Antariksh Narain
 * Description: Calculate the shortest distance for the given map.
 *---------------------------------*/

#include "Const.hpp"

struct Pos
{
    int x;
    int y;
    Pos() {}
    Pos(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
};

struct NodeInfo
{
    Pos idx;
    float conductivity;
    NodeInfo(Pos p, float c)
    {
        this->idx = p;
        this->conductivity = c;
    }
};

class ShortestDistance
{
private:
    const float diag_dist = 1.414;
    const float linear_dist = 1;

    float map[MAP_ROWS][MAP_COLS];

protected:
    float C(Pos pos)
    {
        return map[pos.x][pos.y];
    }
    float Dist(Pos p1, Pos p2)
    {
        if (p1.x != p2.x && p1.y != p2.y)
        {
            return diag_dist;
        }
        return linear_dist;
    }
    NodeInfo NextAdjacent(Pos pos)
    {
        // Using 3-adjacency (x-1,y+1);(x,y+1);(x+1,y+1);
        // Calculate possible positions
        int x = pos.x, y = pos.y;
        vector<Pos> new_pos;
        for (int i = -1; i <= 1; i++)
        {
            if (x + i >= 0 && x + i < MAP_ROWS)
            {
                new_pos.push_back(Pos(x + i, y + 1));
            }
        }
        // Calculate conductivity
        int size = new_pos.size();
        float conductivity[size], total_conduc = 0;
        for (int i = 0; i < size; i++)
        {
            conductivity[i] = C(pos) * C(new_pos[i]) / Dist(pos, new_pos[i]);
            total_conduc += conductivity[i];
        }
        // Compute probability for each adjacent cell
        int max_i = -1;
        float max_prob = -1, prob;
        for (int i = 0; i < size; i++)
        {
            prob = conductivity[i] / total_conduc;
            if (max_prob < prob)
            {
                max_prob = prob;
                max_i = i;
            }
        }
        // return cell with max probability
        return NodeInfo(new_pos[max_i], conductivity[max_i]);
    }

public:
    ShortestDistance(float new_map[MAP_ROWS][MAP_COLS])
    {
        memcpy(this->map, new_map, sizeof(float) * MAP_ROWS * MAP_COLS);
        for(int i=0;i<MAP_ROWS;i++)
        {
            for(int j=0;j<MAP_COLS;j++)
            {
                if(this->map[i][j] != new_map[i][j])
                {
                    printf("%d,%d do not match",i,j);
                    exit(-1);
                }
            }
        }
    }
    vector<NodeInfo> ProbilisticPath(Pos init_position)
    {
        int col = 0;
        vector<NodeInfo> seq;
        seq.push_back(NodeInfo(init_position, C(init_position)));
        while (col < MAP_COLS)
        {
            seq.push_back(this->NextAdjacent(seq[col].idx));
            col++;
        }
        return seq;
    }
    vector<vector<NodeInfo>> ProbilisticPathAll(int rows = MAP_ROWS)
    {
        vector<vector<NodeInfo>> possible_paths;// = vector<vector<NodeInfo>>(rows);
        // PARALLEL
        for (int i = 0; i < rows; i++)
        {
            possible_paths.push_back(this->ProbilisticPath(Pos(i, 0)));
        }
        return possible_paths;
    }
    void DumpPath(vector<vector<NodeInfo>> results)
    {
        fstream ofile("path.csv", ios::out);
        printf("Writing %ld to path.csv\n",results.size());
        for(int i=0;i<results.size();i++)
        {
            //printf("1");
            ofile << results[i][0].idx.x << ":"<< results[i][0].idx.y;
            for(int j=1;j<results[i].size();j++)
            {
                //printf("2");
                ofile << ", " << results[i][j].idx.x << ":"<< results[i][j].idx.y;
            }
            //printf("\n");
            ofile << "\n";
        }
        ofile.close();
    }
};