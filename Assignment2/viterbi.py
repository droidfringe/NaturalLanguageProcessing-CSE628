import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    # print(start_scores.shape)
    dp = np.zeros((L,N),dtype=np.float32)
    parent = np.zeros((L, N), dtype=np.int32)
    dp[:,0] = start_scores + emission_scores[0,:]
    for pos in xrange(1, N):
        for tag in xrange(0,L):
            # Vectorized method
            currScores = dp[:,pos-1] + trans_scores[:,tag]
            currMax = np.max(currScores)
            currMaxParent = np.argmax(currScores)
            # Non-vectorized method
            # currMax = -np.inf
            # prev = -1
            # for prevTag in xrange(0,L):
            #     currScore = dp[prevTag, pos-1] + trans_scores[prevTag, tag]
            #     if(currScore > currMax):
            #         currMax = currScore
            #         prev = prevTag
            dp[tag, pos] = currMax + emission_scores[pos, tag]
            parent[tag, pos] = currMaxParent

    dp[:,N-1] = dp[:,N-1] + end_scores
    score = np.max(dp[:,N-1])
    lastTag = np.argmax(dp[:,N-1])
    y = [lastTag]
    pos = N-1
    # Backtrack on parent path until first position is reached
    for i in xrange(N-1):
        # stupid sequence
        # y.append(i % L)
        lastTag = parent[lastTag][pos]
        pos = pos-1
        y.append(lastTag)
    # score set to 0
    # Reverse the list for the required tag sequence
    y.reverse()
    # Print for debugging purposes
    # print('Emission scores')
    # print(emission_scores)
    # print('Transition scores')
    # print(trans_scores)
    # print('Start scores')
    # print(start_scores)
    # print('End scores')
    # print(end_scores)
    # print('DP table')
    # print(dp)
    # print('Parent table')
    # print(parent)
    # print('Sequence')
    # print(y)
    return (score, y)
