import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, k_list):
    result = []
    for user, item_i, item_j in test_loader:
        pred = model.recommend(user)
        for top_k in k_list:
            HR, NDCG = [], []
            gt_item = item_i[0].item()
            HR.append(hit(gt_item, pred[:top_k]))
            NDCG.append(ndcg(gt_item, pred[:top_k]))
        result.append({"HR":np.mean(HR), "NDCG":np.mean(NDCG)})
    return result



def get_relevance(max_k, user_emb, item_emb, train_user_list, batch):
    # Compute all pair of training and test record
    result = None
    relevance_result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])), value=torch.tensor(0.0))
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        cur_relevance, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
        relevance_result = cur_relevance if relevance_result is None else torch.cat((relevance_result, cur_relevance), dim=0)
    return relevance_result, result


def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, klist, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Calculate max k value
    max_k = max(klist)
    _, result = get_relevance(max_k, user_emb, item_emb, train_user_list, batch)
    
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls