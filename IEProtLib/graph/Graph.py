'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file Graph.py

    \brief Object to represent a graph.

    \copyright Copyright (c) 2021 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf

class Graph:
    """Class to represent a graph.

    Attributes:
        neighbors_ (int tensor n'): List of neighbors (sparse adjacency matrix).
        nodeStartIndexs_ (int tensor n): Index of the starting indices of each node.
    """

    def __init__(self, pNeighbors, pNodeStartIndices):
        """Constructor.

        Args:
            pNeighbors (int tensor n'): List of neighbors (sparse adjacency matrix).
            pNodeStartIndices (int tensor n): Index of the starting indices of each node.
        """
        self.neighbors_ = pNeighbors
        self.nodeStartIndexs_ = pNodeStartIndices
        
        
    def __eq__(self, other):
        """Comparison operator.

        Args:
            other (Graph): Other point cloud.
        Return:
            True if other is equal to self, False otherwise.
        """
        return self.neighbors_.name == other.neighbors_.name and \
            self.nodeStartIndexs_.name == other.nodeStartIndexs_.name


    def __hash__(self):
        """Method to compute the hash of the object.

        Return:
            Unique hash value.
        """
        return hash((self.neighbors_.name, self.nodeStartIndexs_.name))

    
    def pool_graph_drop_nodes(self, pNodeMask, pNewNumNodes):
        """Method to compute the new graph when pooling nodes with dropout nodes.

        Args:
            pNodeMask (bool tensor n): Mask of the nodes to keep.
            pNewNumNodes (int): Number of nodes.
        Return:
            (MCGraph): New pooled graph.
        """
        
        intMask = tf.cast(pNodeMask, tf.int32)

        # Compute the new neighbors.
        accumNeighMask = tf.gather(intMask, self.neighbors_[:, 0]) + \
            tf.gather(intMask, self.neighbors_[:, 1])
        maskNeighbors = tf.equal(accumNeighMask, 2)
        maskedNeighbors = tf.boolean_mask(self.neighbors_, maskNeighbors)

        # Compute the new starting indices.
        index2Remove = 1 - intMask
        index2Remove = tf.math.cumsum(index2Remove)
        auxRange = tf.range(0, tf.shape(pNodeMask)[0], 1)
        newIndices = auxRange - index2Remove
        maskedNeighbors = tf.concat([
            tf.reshape(tf.gather(newIndices, maskedNeighbors[:, 0]), [-1, 1]),
            tf.reshape(tf.gather(newIndices, maskedNeighbors[:, 1]), [-1, 1])],
            axis= 1)
        maskedStartIndices = tf.cast(
            tf.math.unsorted_segment_sum(
            tf.ones_like(maskedNeighbors[:,1]), 
            maskedNeighbors[:,1], 
            pNewNumNodes),
            dtype=tf.int32)
        maskedStartIndices = tf.math.cumsum(maskedStartIndices)

        # Return new graph.
        return Graph(maskedNeighbors, maskedStartIndices)
        
    def pool_graph_collapse_edges(self, pIndices, pNewNumNodes):
        """Method to compute the new graph when pooling nodes with collapsing edges.

        Args:
            pIndices (int tensor n): List of the indices to the new nodes.
            pNewNumNodes (int): Number of nodes.
        Return:
            (Graph): New pooled graph.
        """

        # Create new graph2.
        newNeighColum1 = tf.gather(pIndices, self.neighbors_[:, 0])
        newNeighColum2 = tf.gather(pIndices, self.neighbors_[:, 1])
        _, sortNeighIndexs = tf.math.top_k(newNeighColum2, tf.shape(newNeighColum2)[0])
        sortNeighIndexs = tf.reverse(sortNeighIndexs, [0])
        newNeighColum1 = tf.gather(newNeighColum1, sortNeighIndexs)
        newNeighColum2 = tf.gather(newNeighColum2, sortNeighIndexs)

        # TODO - We are not considering duplicated edges. However, this does not affects
        #  the topo distance computation.
        newNeighbors = tf.concat([
            tf.reshape(newNeighColum1, [-1, 1]),
            tf.reshape(newNeighColum2, [-1, 1])],
            axis= 1)
        maskNeighbors = tf.math.not_equal(newNeighbors[:, 0], newNeighbors[:, 1])
        maskedNewNeighbors = tf.boolean_mask(newNeighbors, maskNeighbors)

        maskedStartIndices = tf.cast(
            tf.math.unsorted_segment_sum(
                tf.ones_like(maskedNewNeighbors[:,1]),
                maskedNewNeighbors[:,1], pNewNumNodes),
            dtype=tf.int32)
        maskedStartIndices = tf.math.cumsum(maskedStartIndices)

        # Return new graph.
        return Graph(maskedNewNeighbors, maskedStartIndices)