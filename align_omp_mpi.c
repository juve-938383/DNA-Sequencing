/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * OpenMP version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.2
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<sys/time.h>
#include<math.h>
#include<omp.h>
#include<mpi.h>

/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX	65535







/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"


/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 *
 */

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence( rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
	unsigned long ind; 
	for( ind=0; ind<length; ind++ ) {
		double prob = rng_next( random );
		if( prob < prob_G ) seq[ind] = 'G';
		else if( prob < prob_C ) seq[ind] = 'C';
		else if( prob < prob_A ) seq[ind] = 'A';
		else seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Copy sample */
	unsigned long ind; 
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
}

/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate( rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length ) {

	/* Random length */
	unsigned long length = (unsigned long)rng_next_normal( random, (double)pat_rng_length_mean, (double)pat_rng_length_dev );
	if ( length > seq_length ) length = seq_length;
	if ( length <= 0 ) length = 1;

	/* Allocate pattern */
	char *pattern = (char *)malloc( sizeof(char) * length );
	if ( pattern == NULL ) {
		fprintf(stderr,"\n-- Error allocating a pattern of size: %lu\n", length );
		exit( EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
}


/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence( rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length ) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Regenerate sample */
	rng_t local_random = random_seq;
	rng_skip( &local_random, location );
	generate_rng_sequence( &local_random, prob_G, prob_C, prob_A, pattern, length);
}


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
	fprintf(stderr,"\n");
}



/*
 * MAIN PROGRAM
 */
int main(int argc, char* argv[]) {
	/* 0. Default output and error without buffering, forces to write immediately */
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);
	int node_count;
	int rank;

	unsigned long seq_length;
	int thread_count;
	unsigned long* pat_length;
	char** pattern;
	int pat_number;
	int patterns_per_node;
	int pattern_per_node_constant;
	int pattern_per_node_root;
	unsigned long* pat_found;
	unsigned long* pat_found_result;
	int* seq_matches;

	int* gather_displs;
	int* gather_rcount;

	unsigned long* misc_data = (unsigned long*)malloc(6 * sizeof(unsigned long));
	if (misc_data == NULL) {
		fprintf(stderr, "\n-- Error allocating misc data structure\n");
		exit(EXIT_FAILURE);
	}

	unsigned long seed;
	rng_t random;
	float prob_G;
	float prob_C;
	float prob_A;
	int ind;
	long slind;
	double ttotal;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &node_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	// Create and commit a contiguous datatype of 10000 characters.
	// To be used in the distribution of patterns.
	MPI_Datatype contiguous_chars_t;
	MPI_Type_contiguous(10000, MPI_CHAR, &contiguous_chars_t);
	MPI_Type_commit(&contiguous_chars_t);

	// The main node sets up the variables, performs the random generation,
	// and distributes the necessary data to the other nodes.
	// The sequence seed is broadcasted to all nodes which generate it locally.
	// The patterns get split (tentatively) equally among the nodes.
	if (rank == 0) {

		/* 1. Read scenary arguments */
		/* 1.1. Check minimum number of arguments */
		if (argc < 15) {
			fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
			show_usage(argv[0]);
			exit(EXIT_FAILURE);
		}

		/* 1.2. Read argument values */
		seq_length = atol(argv[1]);
		prob_G = atof(argv[2]);
		prob_C = atof(argv[3]);
		prob_A = atof(argv[4]);
		if (prob_G + prob_C + prob_A > 1) {
			fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
			show_usage(argv[0]);
			exit(EXIT_FAILURE);
		}
		prob_C += prob_G;
		prob_A += prob_C;

		int pat_rng_num = atoi(argv[5]);
		unsigned long pat_rng_length_mean = atol(argv[6]);
		unsigned long pat_rng_length_dev = atol(argv[7]);

		int pat_samp_num = atoi(argv[8]);
		unsigned long pat_samp_length_mean = atol(argv[9]);
		unsigned long pat_samp_length_dev = atol(argv[10]);
		unsigned long pat_samp_loc_mean = atol(argv[11]);
		unsigned long pat_samp_loc_dev = atol(argv[12]);

		char pat_samp_mix = argv[13][0];
		if (pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M') {
			fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
			show_usage(argv[0]);
			exit(EXIT_FAILURE);
		}

	    seed = atol(argv[14]);

		thread_count = atoi(argv[15]);

#ifdef DEBUG
		/* DEBUG: Print arguments */
		printf("\nArguments: seq_length=%lu\n", seq_length);
		printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A);
		printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev);
		printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev);
		printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed);
		printf("\n");
#endif // DEBUG

		/* 2. Initialize data structures */
		/* 2.1. Skip allocate and fill sequence */
		random = rng_new(seed);
		rng_skip(&random, seq_length);

		/* 2.2. Allocate and fill patterns */
		/* 2.2.1 Allocate main structures */
		pat_number = pat_rng_num + pat_samp_num;
		pat_length = (unsigned long*)malloc(sizeof(unsigned long) * pat_number);
		pattern = (char**)malloc(sizeof(char*) * pat_number);
		if (pattern == NULL || pat_length == NULL) {
			fprintf(stderr, "\n-- Error allocating the basic patterns structures for size: %d\n", pat_number);
			exit(EXIT_FAILURE);
		}

		/* 2.2.2 Allocate and initialize ancillary structure for pattern types */
		
#define PAT_TYPE_NONE	0
#define PAT_TYPE_RNG	1
#define PAT_TYPE_SAMP	2
		char* pat_type = (char*)malloc(sizeof(char) * pat_number);
		if (pat_type == NULL) {
			fprintf(stderr, "\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number);
			exit(EXIT_FAILURE);
		}
		for (ind = 0; ind < pat_number; ind++) pat_type[ind] = PAT_TYPE_NONE;

		/* 2.2.3 Fill up pattern types using the chosen mode */
		switch (pat_samp_mix) {
		case 'A':
			for (ind = 0; ind < pat_rng_num; ind++) pat_type[ind] = PAT_TYPE_RNG;
			for (; ind < pat_number; ind++) pat_type[ind] = PAT_TYPE_SAMP;
			break;
		case 'B':
			for (ind = 0; ind < pat_samp_num; ind++) pat_type[ind] = PAT_TYPE_SAMP;
			for (; ind < pat_number; ind++) pat_type[ind] = PAT_TYPE_RNG;
			break;
		default:
			if (pat_rng_num == 0) {
				for (ind = 0; ind < pat_number; ind++) pat_type[ind] = PAT_TYPE_SAMP;
			}
			else if (pat_samp_num == 0) {
				for (ind = 0; ind < pat_number; ind++) pat_type[ind] = PAT_TYPE_RNG;
			}
			else if (pat_rng_num < pat_samp_num) {
				int interval = pat_number / pat_rng_num;
				for (ind = 0; ind < pat_number; ind++)
					if ((ind + 1) % interval == 0) pat_type[ind] = PAT_TYPE_RNG;
					else pat_type[ind] = PAT_TYPE_SAMP;
			}
			else {
				int interval = pat_number / pat_samp_num;
				for (ind = 0; ind < pat_number; ind++)
					if ((ind + 1) % interval == 0) pat_type[ind] = PAT_TYPE_SAMP;
					else pat_type[ind] = PAT_TYPE_RNG;
			}
		}

		/* 2.2.4 Generate the patterns */
		for (ind = 0; ind < pat_number; ind++) {
			if (pat_type[ind] == PAT_TYPE_RNG) {
				pattern[ind] = pattern_allocate(&random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind]);
				generate_rng_sequence(&random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind]);
			}
			else if (pat_type[ind] == PAT_TYPE_SAMP) {
				pattern[ind] = pattern_allocate(&random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind]);
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
				rng_t random_seq_orig = rng_new(seed);
				generate_sample_sequence(&random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#else
				copy_sample_sequence(&random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind]);
#endif
			}
			else {
				fprintf(stderr, "\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind);
				exit(EXIT_FAILURE);
			}
		}
		free(pat_type);

		/* Avoid the usage of arguments to take strategic decisions
		 * In a real case the user only has the patterns and sequence data to analize
		 */
		argc = 0;
		argv = NULL;
		pat_rng_num = 0;
		pat_rng_length_mean = 0;
		pat_rng_length_dev = 0;
		pat_samp_num = 0;
		pat_samp_length_mean = 0;
		pat_samp_length_dev = 0;
		pat_samp_loc_mean = 0;
		pat_samp_loc_dev = 0;
		pat_samp_mix = '0';




		// All nodes but node 0 get an equal share of patterns to check
		// Node 0 gets that amount plus the remainder
		patterns_per_node = (int)floor(pat_number / (float)node_count);
		int patterns_for_root_node = pat_number - patterns_per_node * (node_count - 1);
		pattern_per_node_constant = patterns_per_node;
		misc_data[0] = seq_length;
		misc_data[1] = (unsigned long)thread_count;
		misc_data[2] = (unsigned long)patterns_per_node;
		misc_data[3] = (unsigned long)pat_number;
		misc_data[4] = (unsigned long)node_count;
		patterns_per_node = patterns_for_root_node;
		pattern_per_node_root = patterns_per_node;
		misc_data[5] = (unsigned long)patterns_per_node;


	}

	// Broadcast some auxiliary data
	MPI_Bcast(misc_data, 6, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);


	if (rank != 0) {
		// Receiving and assigning the data
		seq_length = misc_data[0];
		thread_count = (int)misc_data[1];
		patterns_per_node = (int)misc_data[2];
		pat_number = (int)misc_data[3];
		node_count = (int)misc_data[4];
		pattern_per_node_root = (int)misc_data[5];
		pat_length = (unsigned long*)malloc(sizeof(unsigned long) * patterns_per_node);
		pattern = (char**)malloc(sizeof(char*) * patterns_per_node);
		if (pat_length == NULL || pattern == NULL) {
			fprintf(stderr, "\n-- Error allocating pattern structure\n");
			exit(EXIT_FAILURE);
		}
		
	}

	pat_found = (unsigned long*)malloc(sizeof(unsigned long) * patterns_per_node);
	if (pat_found == NULL) {
		fprintf(stderr, "\n-- Error allocating aux pattern structure\n");
		exit(EXIT_FAILURE);
	}

	pat_found_result = (unsigned long*)malloc(sizeof(unsigned long) * pat_number);

	if (pat_found_result == NULL) {
		fprintf(stderr, "\n-- Error allocating aux pattern result structure on rank %d\n", rank);
		exit(EXIT_FAILURE);
	}

	gather_displs = (int*)malloc(node_count * sizeof(int));
	gather_rcount = (int*)malloc(node_count * sizeof(int));
	if (gather_displs == NULL || gather_rcount == NULL) {
		fprintf(stderr, "\n-- Error allocating aux gather structures on rank %d\n", rank);
		exit(EXIT_FAILURE);
	}

	/* 2.3. Other result data and structures */
	int pat_matches = 0;
	seq_matches = (int*)malloc(sizeof(int) * seq_length);
	if (seq_matches == NULL) {
		fprintf(stderr, "\n-- Error allocating aux sequence structures for size: %lu\n", seq_length);
		exit(EXIT_FAILURE);
	}

	if (rank == 0) {
		// Creating and assigning auxiliary structures for the distribution of patterns and their lengths
		int* scounts_lens = (int*)malloc(sizeof(int) * node_count);
		int* displs_lens = (int*)malloc(sizeof(int) * node_count);
		int* scounts_chars = (int*)malloc(sizeof(int) * node_count);
		int* displs_chars = (int*)malloc(sizeof(int) * node_count);
		unsigned long* char_counts = (unsigned long*)malloc(sizeof(unsigned long) * node_count);

		if(scounts_lens == NULL){
			fprintf(stderr, "\n-- Error allocating scounts_lens");
			exit(EXIT_FAILURE);
		}
		if (displs_lens == NULL) {
			fprintf(stderr, "\n-- Error allocating displs_lens");
			exit(EXIT_FAILURE);
		}
		if (scounts_chars == NULL) {
			fprintf(stderr, "\n-- Error allocating scounts_chars");
			exit(EXIT_FAILURE);
		}
		if (displs_chars == NULL) {
			fprintf(stderr, "\n-- Error allocating displs_chars");
			exit(EXIT_FAILURE);
		}
		if (char_counts == NULL) {
			fprintf(stderr, "\n-- Error allocating char_counts");
			exit(EXIT_FAILURE);
		}

		int index = 0;
		int increment = patterns_per_node;
		for (int i = 0; i < node_count; i++) {
			char_counts[i] = 0;
#pragma omp parallel for reduction(+: char_counts[i])
			for (int j = index; j < index + increment; j++) {
				char_counts[i] += pat_length[j];
			}

			index += increment;
			increment = pattern_per_node_constant;
		}

		scounts_lens[0] = patterns_per_node;
		displs_lens[0] = 0;
		scounts_chars[0] = (int)ceil(char_counts[0] / 10000.0f);
		displs_chars[0] = 0;
		unsigned long total_char_count = 10000ul * scounts_chars[0];

		for (int i = 1; i < node_count; i++) {
			scounts_lens[i] = pattern_per_node_constant;
			displs_lens[i] = displs_lens[i-1] + scounts_lens[i-1];
			scounts_chars[i] = (int)ceil(char_counts[i] / 10000.0f);
			displs_chars[i] = displs_chars[i - 1] + scounts_chars[i - 1];
			total_char_count += 10000ul * scounts_chars[i];
		}
		// Distributing the pattern lengths using scounts_lens to indicate the number that goes to each node.
		// Reminder: All nodes get the same amount of patterns; node 0 gets that amount plus the remainder.
		MPI_Scatterv(pat_length, scounts_lens, displs_lens, MPI_UNSIGNED_LONG, MPI_IN_PLACE, scounts_lens[0], MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
		
		// Creating a contiguous array of all patterns to distribute among the nodes.
		char* cont_chars_array = (char*)malloc(sizeof(char) * total_char_count);
		if (cont_chars_array == NULL) {
			fprintf(stderr, "\n-- Error allocating send cont_chars_array");
			exit(EXIT_FAILURE);
		}

		int pattern_index = 0;
		unsigned long char_index = 0;
		int pattern_limit = patterns_per_node;
		unsigned long loop_index = 0;
		for (int i = 0; i < node_count; i++) {
			for (int j = 0; j < scounts_chars[i]; j++) {
				for (int k = 0; k < 10000; k++) {
					if (pattern_index == pattern_limit) {
						// Some padding is necessary if the number of characters of all patterns of a node is not a multiple of 10,000
						cont_chars_array[loop_index] = '.';
						loop_index++;
						continue;
					}


					cont_chars_array[loop_index] = pattern[pattern_index][char_index];
					char_index++;
					if (char_index == pat_length[pattern_index]) {
						pattern_index++;
						char_index = 0ul;
					}
					loop_index++;
				}
			}
			pattern_limit += pattern_per_node_constant;
		}
		// Scattering the characters
		MPI_Scatterv(cont_chars_array, scounts_chars, displs_chars, contiguous_chars_t, MPI_IN_PLACE, scounts_chars[0], contiguous_chars_t, 0, MPI_COMM_WORLD);
		
		free(cont_chars_array);
		free(scounts_lens);
		free(displs_lens);
		free(scounts_chars);
		free(displs_chars);
		free(char_counts);

		// Preparing the data needed for sequence generation to be distributed
		unsigned long* temp = (unsigned long*) &prob_G;
		misc_data[0] = seed;
		misc_data[1] = *temp;
		temp = (unsigned long*)&prob_C;
		misc_data[2] = *temp;
		temp = (unsigned long*)&prob_A;
		misc_data[3] = *temp;
		misc_data[4] = seq_length;



		/* 3. Start global timer */
		ttotal = cp_Wtime();

#ifdef DEBUG
		/* DEBUG: Print sequence and patterns */
		printf("-----------------\n");
		printf("Sequence: ");
		for (lind = 0; lind < seq_length; lind++)
			printf("%c", sequence[lind]);
		printf("\n-----------------\n");
		printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num);
		int debug_pat;
		for (debug_pat = 0; debug_pat < pat_number; debug_pat++) {
			printf("Pat[%d]: ", debug_pat);
			for (lind = 0; lind < pat_length[debug_pat]; lind++)
				printf("%c", pattern[debug_pat][lind]);
			printf("\n");
		}
		printf("-----------------\n\n");
#endif // DEBUG

		/* 2.3.2. Other results related to the main sequence */
		
		
	}
	else {

		// Recieving and assigning the respective values.
		MPI_Scatterv(NULL, NULL, NULL, MPI_UNSIGNED_LONG, pat_length, patterns_per_node, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

		unsigned long total_char_count = 0;
#pragma omp parallel for reduction(+: total_char_count)
		for (int i = 0; i < patterns_per_node; i++) {
			total_char_count += pat_length[i];
		}


		int subarrayCount = (int)ceil(total_char_count / 10000.0f);
		char* cont_chars_array = (char*)malloc(sizeof(char) * subarrayCount * 10000ul);
		if (cont_chars_array == NULL) {
			fprintf(stderr, "\n-- Error allocating recieve cont_chars_array");
			exit(EXIT_FAILURE);
		}
		
		// Recieving and storing the patterns
		MPI_Scatterv(NULL, NULL, NULL, contiguous_chars_t, cont_chars_array, subarrayCount, contiguous_chars_t, 0, MPI_COMM_WORLD);
	

		for (int i = 0; i < patterns_per_node; i++) {
			pattern[i] = (char*)malloc(pat_length[i] * sizeof(char));
		}


		int pattern_index = 0;
		unsigned long char_index = 0ul;

		for (int j = 0; j < subarrayCount; j++) {
			for (int k = 0; k < 10000; k++) {
				if (pattern_index == patterns_per_node)
					goto exit_loop;
				pattern[pattern_index][char_index] = cont_chars_array[j * 10000ul + k];
				char_index++;
				if (char_index == pat_length[pattern_index]) {
					pattern_index++;
					char_index = 0ul;
				}
			}
		}
exit_loop:
		free(cont_chars_array);

	}

	char* sequence = (char*)malloc(sizeof(char) * seq_length);
	if (sequence == NULL) {
		fprintf(stderr, "\n-- Error allocating the sequence for size: %lu\n", seq_length);
		exit(EXIT_FAILURE);
	}


	
	// Broadcasting the sequence data to all nodes
	MPI_Bcast(misc_data, 5, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

	if (rank == 0) {

#pragma omp parallel for
		for (slind = 0; slind < seq_length; slind++) {
			seq_matches[slind] = NOT_FOUND;
		}

		
	}else{

		seed = misc_data[0];
		unsigned long temp_ul = misc_data[1];
		float* temp_f = (float*) &temp_ul;
		prob_G = *temp_f;
		temp_ul = misc_data[2];
		temp_f = (float*) &temp_ul;
		prob_C = *temp_f;
		temp_ul = misc_data[3];
		temp_f = (float*)&temp_ul;
		prob_A = *temp_f;
		seq_length = misc_data[4];


	}


#pragma omp parallel for
	for (ind = 0; ind < patterns_per_node; ind++) {
		pat_found[ind] = (unsigned long)NOT_FOUND;
	}


	// Sequence generation
	random = rng_new(seed);
	generate_rng_sequence(&random, prob_G, prob_C, prob_A, sequence, seq_length);



	// The main loop of the algorithm only writes 
	// the indices where each pattern was found in the respective local buffer.
	// The calculations for the final results are performed in the end 
	// by node 0 once it has collected these buffers.

	/* 5. Search for each pattern */
	int pat_ind;
#pragma omp parallel for num_threads(thread_count) schedule(static, patterns_per_node/thread_count)
	for( pat_ind=0; pat_ind < patterns_per_node; pat_ind++ ) {
		
		/* 5.1. For each posible starting position */
		unsigned long start;
		for( start=0; start <= seq_length - pat_length[pat_ind]; start++) {
			unsigned long i;
			/* 5.1.1. For each pattern element */
			for( i=0; i<pat_length[pat_ind]; i++) {
				/* Stop this test when different nucleotids are found */
				if ( sequence[start + i] != pattern[pat_ind][i] ) break;
			}
			/* 5.1.2. Check if the loop ended with a match */
			if ( i == pat_length[pat_ind] ) {
				pat_found[pat_ind] = start;
				break;
			}
		}

	}

	// All other nodes send their results to node 0
		
	gather_displs[0] = 0;
	gather_rcount[0] = pattern_per_node_root;
	int temp = rank == 0 ? pattern_per_node_constant : patterns_per_node;
	for (int m = 1; m < node_count; m++) {
		gather_displs[m] = pattern_per_node_root + (m - 1) * temp;
		gather_rcount[m] = temp;
	}

	MPI_Gatherv(pat_found, patterns_per_node, MPI_UNSIGNED_LONG, pat_found_result, gather_rcount, gather_displs, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
	

	// Node 0 collects the results of each node, computes the final results and outputs them.
	if (rank == 0) {

#pragma omp parallel for num_threads(thread_count)  reduction(+: seq_matches[:seq_length]) reduction(+: pat_matches) schedule(static, pat_number/thread_count)
		for (pat_ind = 0; pat_ind < pat_number; pat_ind++) {
			/* 5.2. Pattern found */
			if (pat_found_result[pat_ind] != (unsigned long)NOT_FOUND) {
				pat_matches++;
				/* 4.2.1. Increment the number of pattern matches on the sequence positions */
				unsigned long i;
				for (i = 0; i < pat_length[pat_ind]; i++) {
					//#pragma omp atomic
					seq_matches[pat_found_result[pat_ind] + i]++;
				}
			}
		}


		/* 7. Check sums */
		unsigned long checksum_matches = 0;
		unsigned long checksum_found = 0;
#pragma omp parallel for reduction(+: checksum_found)
		for (ind = 0; ind < pat_number; ind++) {
			if (pat_found_result[ind] != (unsigned long)NOT_FOUND)
				checksum_found = (checksum_found + pat_found_result[ind]) % CHECKSUM_MAX;
		}
#pragma omp parallel for reduction(+: checksum_matches)
		for (slind = 0; slind < seq_length; slind++) {
			if (seq_matches[slind] != NOT_FOUND)
				checksum_matches = (checksum_matches + seq_matches[slind]) % CHECKSUM_MAX;
		}

		checksum_found = checksum_found % CHECKSUM_MAX;
		checksum_matches = checksum_matches % CHECKSUM_MAX;




#ifdef DEBUG
		/* DEBUG: Write results */
		printf("-----------------\n");
		printf("Found start:");
		for (debug_pat = 0; debug_pat < pat_number; debug_pat++) {
			printf(" %lu", pat_found[debug_pat]);
		}
		printf("\n");
		printf("-----------------\n");
		printf("Matches:");
		for (lind = 0; lind < seq_length; lind++)
			printf(" %d", seq_matches[lind]);
		printf("\n");
		printf("-----------------\n");
#endif // DEBUG

		/* Free local resources */

		/*
		 *
		 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
		 *
		 */

		 /* 8. Stop global timer */
		ttotal = cp_Wtime() - ttotal;

		/* 9. Output for leaderboard */
		printf("\n");
		/* 9.1. Total computation time */
		printf("Time: %lf\n", ttotal);

		/* 9.2. Results: Statistics */
		printf("Result: %d, %lu, %lu\n\n",
			pat_matches,
			checksum_found,
			checksum_matches);

		free(pat_found_result);
	}
	MPI_Type_free(&contiguous_chars_t);
	/* 10. Free resources */	
	MPI_Finalize();
	free(sequence);
	free(seq_matches);
	int i;
	if(rank==0)
		for( i=0; i<pat_number; i++ ) free( pattern[i] );
	else
		for( i=0; i<patterns_per_node; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );
	free(misc_data);
	free(gather_displs);
	free(gather_rcount);

	/* 11. End */
	return 0;
}
