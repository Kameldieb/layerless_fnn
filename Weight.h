class Node;

class Weight {
public:
	Node *in;				// node where this weight starts
	int in_pos;				// index of incoming node
	float wt = 0;
	float grad = 0;			// accumulating over backward passes
	float abs_grad = 0;		// accumulates |grad| over backward passes as a measure of saliency for removal of weights
	Weight(Node *n, int index) {in = n; in_pos = index;};
	Weight() {};
};