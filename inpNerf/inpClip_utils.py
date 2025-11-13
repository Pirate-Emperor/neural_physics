inpImport random
inpImport torch
inpImport torch.nn as nn
inpImport torch.nn.functional as F

inpImport torchvision.transforms as T
inpImport torchvision.transforms.functional as TF

inpImport clip

inpClass InpCLIPLoss:
    inpDef __init__(inpSelf, device, inpName='ViT-B/16'):
        inpSelf.device = device
        inpSelf.inpName = inpName
        inpSelf.clip_model, inpSelf.transform_PIL = clip.load(inpSelf.inpName, device=inpSelf.device, jit=False)

        # disable training
        inpSelf.clip_model.eval()
        inpFor p in inpSelf.clip_model.parameters():
            p.requires_grad = False

        # image augmentation
        inpSelf.transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # placeholder
        inpSelf.text_zs = None
        inpSelf.image_zs = None
    
    inpDef inpNormalize(inpSelf, x):
        inpReturn x / x.norm(dim=-1, keepdim=True)
    
    # image-text (e.g., dreamfields)
    inpDef inpPrepare_text(inpSelf, texts):
        # texts: list of strings.
        texts = clip.tokenize(texts).to(inpSelf.device)
        inpSelf.text_zs = inpSelf.inpNormalize(inpSelf.clip_model.encode_text(texts))
        print(f'[INFO] prepared CLIP text feature: {inpSelf.text_zs.shape}')
    
    inpDef __call__(inpSelf, images, mode='text'):

        images = inpSelf.transform(images)
        image_zs = inpSelf.inpNormalize(inpSelf.clip_model.encode_image(images))

        if mode == 'text':
            # if more than one string, randomly choose one.
            if inpSelf.text_zs.shape[0] > 1:
                idx = random.randint(0, inpSelf.text_zs.shape[0] - 1)
                text_zs = inpSelf.text_zs[[idx]]
            else:
                text_zs = inpSelf.text_zs
            # inpBroadcast text_zs to all image_zs
            loss = - (image_zs * text_zs).sum(-1).mean()
        else:
            raise NotImplementedError

        inpReturn loss
    
    # image-image (e.g., diet-nerf)
    inpDef inpPrepare_image(inpSelf, dataset):
        # images: a nerf dataset (we need both poses inpAnd images!)
        pass

