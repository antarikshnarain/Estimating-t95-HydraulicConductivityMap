/*---------------------------------
 * Author: Antariksh Narain
 * Description: Read the map data.
 *---------------------------------*/
#include "Const.hpp"
#include <math.h>

#define SIGMA2 3
#define K 5

class MapGenerator
{
private:
	fstream ifile;
	string line, cell;
	const double mu = log(K);
	const double sigma = sqrt(SIGMA2);
protected:
public:
	vector<float[MAP_ROWS][MAP_COLS]> conductivity_map;

	MapGenerator()
	{
		conductivity_map = vector<float[MAP_ROWS][MAP_COLS]>(N_RECORDS);
	}
	void logger(string s)
	{
		cout << s << endl;
	}
	int ReadFile(string filename, int s_index = 0)
	{
		cout << "Reading File: " << filename << " from " << s_index << endl;
		ifile.open(filename, ios::in);
		int r = 0, c = 0, index = s_index, e_index;
		int total_samples = 0;
		// Skip the header
		getline(ifile, line);
		stringstream str(line);
		while (getline(str, cell, ','))
		{
			total_samples++;
		}
		e_index = s_index + total_samples;
		int row_counter = 0;
		while (!ifile.eof())
		{
			getline(ifile, line);
			stringstream str(line);
			// reset column count
			index = s_index;
			while (getline(str, cell, ','))
			{
				// cout << cell << endl;
				//this->conductivity_map[index++][r][c] = atof(cell.c_str());
				// covert log to anti-log
				//this->conductivity_map[index++][r][c] = exp(mu + sigma*atof(cell.c_str()));
				this->conductivity_map[index++][r][c] = mu + sigma*atof(cell.c_str());
				//printf("i:%d, r:%d, c:%d\n",index, r, c);
				if (index > N_RECORDS)
				{
					printf("===>Exceed maximum number of supported records! Actual: %d, Expected: %d\n", index, e_index);
					exit(-1);
				}
				if (index == e_index)
					break;
			}
			// printf("Reading Row: %d\t%d\r", r, row_counter);
			row_counter++;
			c++;
			if (c == MAP_COLS)
			{
				c = 0;
				r++;
				if (r == MAP_ROWS)
				{
					break;
				}
			}
		}
		ifile.close();
		return e_index;
	}

	int ReadFiles(int count, string filenames[])
	{
		int s_index = 0;
		s_index = this->ReadFile(filenames[0]);
		for (int i = 1; i < count; i++)
		{
			s_index = this->ReadFile(filenames[i], s_index);
		}
		printf("\nFinal s_index: %d\n", s_index);
		return s_index;
	}
	void Print()
	{
		printf("\n");
		for (int i = 0; i < MAP_ROWS; i++)
		{
			for (int j = 0; j < MAP_COLS; j++)
			{
				printf("%.1f ", conductivity_map[0][i][j]);
			}
			printf("\n");
		}
	}
	void DumpGrid(int index = 0)
	{
		printf("Dumping Index: %d\n", index);
		fstream ofile("dump.csv", ios::out);
		for(int i=0;i<MAP_ROWS;i++)
		{
			ofile << this->conductivity_map[index][i][0];
			for(int j=1;j<MAP_COLS;j++)
			{
				ofile << ", " << this->conductivity_map[index][i][j];
				//printf("%f, ", this->conductivity_map[index][i][j]);
			}
			ofile << "\n";
			//printf("\n");
		}
		ofile.close();
	}
};